#!/usr/bin/env python3
"""
Diagnose CockroachDB connection issues.
Safely parses URL and tests TCP connectivity without exposing credentials.
"""

import os
import socket
from urllib.parse import urlparse, parse_qs


def diagnose_connection():
    """Parse connection URL and test TCP connectivity."""

    # Get URL from environment
    url = os.environ.get('COCKROACHDB_URL')

    if not url:
        print("❌ COCKROACHDB_URL not set")
        return

    print("✓ COCKROACHDB_URL is set")
    print()

    # Parse URL
    try:
        parsed = urlparse(url)
        print("✓ URL structure valid")
        print()

        # Extract components (safe to display)
        scheme = parsed.scheme
        hostname = parsed.hostname
        port = parsed.port or 26257  # CockroachDB default
        username = parsed.username
        database = parsed.path.lstrip('/') if parsed.path else ''

        # Parse query parameters for sslmode
        query_params = parse_qs(parsed.query)
        sslmode = query_params.get('sslmode', [''])[0] or query_params.get('ssl', [''])[0]

        # Display parsed components
        print("=== Connection URL Components ===")
        print(f"Scheme: {scheme}")
        print(f"Hostname: {hostname}")
        print(f"Port: {port}")
        print(f"Username: {username}")
        print(f"Database: {database}")
        print(f"SSL Mode: {sslmode if sslmode else '(not specified)'}")
        print()

    except Exception as e:
        print(f"❌ URL structure invalid: {type(e).__name__}")
        return

    # Test TCP connectivity
    print("=== TCP Connectivity Test ===")
    print(f"Testing connection to {hostname}:{port}...")

    try:
        sock = socket.create_connection((hostname, port), timeout=10)
        sock.close()
        print(f"✓ TCP connection successful to {hostname}:{port}")

    except Exception as e:
        error_type = type(e).__name__
        print(f"❌ TCP connection failed: {error_type}")
        print(f"   (Unable to establish TCP socket to {hostname}:{port})")


if __name__ == '__main__':
    diagnose_connection()
