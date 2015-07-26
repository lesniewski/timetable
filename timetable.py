"""
Data wrangling for nextbus data analysis.
"""

import argparse
import bz2
from collections import Iterable, namedtuple
from datetime import datetime
import gzip
import logging
import numpy as np
import os
import subprocess
import sys

vehicle_location = namedtuple('vehicle_location', 'timestamp id route dir lat lon heading speed')

def read_csv(*csv_files):
    header_fields = list(vehicle_location._fields)
    for f in csv_files:
        with f:
            for line in f:
                fields = line.rstrip('\n').split(',')
                if fields == header_fields:
                    continue  # skip header line
                else:
                    loc = vehicle_location(*fields)
                    yield vehicle_location(
                            timestamp=float(loc.timestamp),
                            id=int(loc.id),
                            route=str(loc.route),
                            dir=(str(loc.dir) or None),
                            lat=float(loc.lat),
                            lon=float(loc.lon),
                            heading=int(loc.heading),
                            speed=int(loc.speed),
                            )

class Archive(object):
    def __init__(self, f):
        self.previous_locations = {}
        self.subprocess = None
        if isinstance(f, file):
            self.fs = [f]
        elif isinstance(f, Iterable) and not isinstance(f, basestring):
            self.fs = (Archive(x) for x in f)
        elif os.path.isdir(f):
            self.fs = (Archive(os.path.join(f, filename)) for filename in sorted(os.listdir(f)))
        elif f.endswith(('.tar', '.tar.gz', '.tar.bz2')):
            self.subprocess = subprocess.Popen(['tar', '-xaOf', f], stdout=subprocess.PIPE)
            self.fs = [self.subprocess.stdout]
        else:
            assert '.' not in f or f.endswith(('.gz', '.bz2'))
            self.fs = [open_maybe_compressed(f)]

    def __iter__(self):
        for f in self.fs:
            with f:
                for line in f:
                    yield line

    def readlines(self):
        return iter(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self):
        for f in self.fs:
            f.close()
        if self.subprocess:
            assert self.subprocess.wait() == 0

    def csv(self):
        yield ','.join(vehicle_location._fields) + '\n'
        for loc in self.locations():
            yield ','.join('' if x is None else str(x) for x in loc) + '\n'

    def locations(self):
        for query_timestamp, vehicle_locations in sorted(self.query_data()):
            for loc in sorted(vehicle_locations):
                # Check whether we've seen this location recently. Because of races in nextbus's
                # servers, timestamps are sometimes skewed by up to 3 seconds. If the timestamp differs
                # by less than 3 seconds, we'll consider it the same data point. If the timestamp is
                # more than 3 seconds later, we'll consider the vehicle to have been stationary.
                prevs = self.previous_locations.setdefault(loc.id, [])
                if any(abs(loc.timestamp - prev.timestamp) < 3 and
                       loc.route == prev.route and
                       abs(loc.lat - prev.lat) < 0.0000001 and
                       abs(loc.lon - prev.lon) < 0.0000001 and
                       loc.speed == prev.speed and
                       (loc.heading == prev.heading or loc.speed == 0)
                       for prev in prevs):
                    continue

                if prevs and loc.timestamp <= prevs[-1].timestamp:
                    logging.warning('Out of order timestamp at query %s: '
                            'id="%i" routeTag="%s" dirTag="%s" lat="%.7f" lon="%.7f" '
                            'secsSinceReport="%i" predictable="true" heading="%i" speedKmHr="%i"',
                            datetime.fromtimestamp(query_timestamp),
                            loc.id, loc.route, loc.dir, loc.lat, loc.lon,
                            query_timestamp - loc.timestamp, loc.heading, loc.speed)

                yield loc

                # Track the previous 3 locations. Sometimes we get a stale data point from nextbus.
                prevs.append(loc)
                if len(prevs) > 3:
                    prevs.pop(0)

    def query_data(self):
        for body in self.bodies():
            # Check and discard truncated bodies, keep only the last, full one.
            xml_header = [i for i, line in enumerate(body)
                          if line.strip().endswith('<?xml version="1.0" encoding="utf-8" ?>')]
            body = body[xml_header[-1]:]

            assert body[1].strip().startswith('<body copyright="'), body[1]
            assert body[1].strip().endswith('">'), body[1]
            assert body[-1].strip() == '</body>', body[-1]

            if body[2].strip().startswith('<Error shouldRetry="'):
                assert body[-2].strip() == '</Error>', body[-2]
                continue

            assert body[-2].strip().startswith('<lastTime time="'), body[-2]
            assert body[-2].strip().endswith('"/>'), body[-2]
            _, last_time_str, _ = body[-2].split('"')
            last_time = 0.001 * int(last_time_str)

            vehicle_locations = {}
            for line in body[2:-2]:
                assert line.strip().startswith('<vehicle id="') and line.strip().endswith('"/>'), line
                attrs = {}
                for attr in line.strip()[len('<vehicle '):-len('/>')].split():
                    name, value = attr.split('=')
                    assert name not in attrs, line
                    assert value[0] == value[-1] == '"', line
                    attrs[name] = value[1:-1]

                loc = vehicle_location(
                    timestamp=int(attrs['secsSinceReport']),
                    id=int(attrs['id']),
                    route=attrs['routeTag'],
                    dir=attrs.get('dirTag'),
                    lat=float(attrs['lat']),
                    lon=float(attrs['lon']),
                    heading=int(attrs['heading']),
                    speed=int(attrs['speedKmHr']),
                    )
                assert loc.id not in vehicle_locations, line
                vehicle_locations[loc.id] = loc

            if vehicle_locations:
                query_timestamp = last_time + min(loc.timestamp for loc in vehicle_locations.values())
                yield query_timestamp, {
                        loc._replace(timestamp=query_timestamp - loc.timestamp)
                        for loc in vehicle_locations.values() }

    def bodies(self):
        body = []
        for line in self.readlines():
            body.append(line)

            if body[0].strip() == '<html>':
                # Discard error pages.
                if line.strip() == '</html>':
                    body = []

            elif line.strip() == '</body>':
                yield body
                body = []


def open_maybe_compressed(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    elif filename.endswith('.bz2'):
        return bz2.BZ2File(filename, mode, buffering=100000)
    else:
        return open(filename, mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logfile', type=str)
    sp = ap.add_subparsers(dest='command')
    csv = sp.add_parser('csv')
    csv.add_argument('archives', nargs='+')
    read = sp.add_parser('read')
    read.add_argument('filename')
    args = ap.parse_args()
    logging.basicConfig(filename=args.logfile)
    if args.command == 'csv':
        with Archive(args.archives) as archive:
            sys.stdout.writelines(archive.csv())
    elif args.command == 'read':
        sys.stdout.writelines(str(loc)+'\n' for loc in read_csv(open_maybe_compressed(args.filename)))
    else:
        assert False

if __name__ == '__main__':
    main()
