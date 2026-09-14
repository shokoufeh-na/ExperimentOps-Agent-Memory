#!/usr/bin/env python3
"""
Safely inspect COCKROACHDB_URL structure.
Reports connection parameters without exposing credentials.
"""

import os
from urllib.parse import urlparse, parse_qs


def inspect_url():
    """Parse and inspect connection URL."""

    url = os.environ.get('COCKROACHDB_URL')

    if not url:
        print("❌ COCKROACHDB_URL not set")
        return

    try:
        parsed = urlparse(url)

        # Extract components
        scheme = parsed.scheme
        hostname = parsed.hostname
        port = parsed.port or 26257
        username = parsed.username
        password = parsed.password
        database = parsed.path.lstrip('/') if parsed.path else ''

        # Parse query parameters
        query_params = parse_qs(parsed.query)
        sslmode = query_params.get('sslmode', [''])[0] or query_params.get('ssl', [''])[0]
        sslrootcert = query_params.get('sslrootcert', [''])[0]

        # Display safe information
        print("=== URL Components ===")
        print(f"Scheme: {scheme}")
        print(f"Hostname: {hostname}")
        print(f"Port: {port}")
        print(f"Username: {username}")
        print(f"Database: {database}")
        print(f"SSL Mode: {sslmode if sslmode else '(not specified)'}")
        print()

        # SSL certificate configuration
        print("=== SSL Certificate ===")
        if sslrootcert:
            print(f"sslrootcert present: {sslrootcert}")
        else:
            print("sslrootcert: (not present)")
        print()

        # Password presence
        print("=== Password ===")
        print(f"Password component exists: {bool(password)}")

    except Exception as e:
        print(f"❌ Error parsing URL: {type(e).__name__}")


if __name__ == '__main__':
    inspect_url()
