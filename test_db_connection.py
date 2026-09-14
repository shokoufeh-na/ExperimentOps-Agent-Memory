#!/usr/bin/env python3
"""
Test CockroachDB connectivity through the Python application.
Read-only operations only.
"""

import os
import sys

# Add lambda directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lambda'))

from db import CockroachDBClient


def test_connection():
    """Test database connection with read-only queries."""

    # Check if COCKROACHDB_URL is set (but don't print it)
    url_set = 'COCKROACHDB_URL' in os.environ
    print(f"COCKROACHDB_URL set: {url_set}")

    if not url_set:
        print("❌ Connection failed: COCKROACHDB_URL not set")
        return

    try:
        # Create client and connect
        with CockroachDBClient() as db:
            print("✓ Connection successful")

            # Query current database
            with db.conn.cursor() as cur:
                cur.execute("SELECT current_database()")
                db_name = cur.fetchone()['current_database']
                print(f"Current database: {db_name}")

            # Query current user
            with db.conn.cursor() as cur:
                cur.execute("SELECT current_user")
                user = cur.fetchone()['current_user']
                print(f"Current user: {user}")

            # Count experiments
            with db.conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) as count FROM experiments")
                exp_count = cur.fetchone()['count']
                print(f"experiments row count: {exp_count}")

            # Count agent_memories
            with db.conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) as count FROM agent_memories")
                mem_count = cur.fetchone()['count']
                print(f"agent_memories row count: {mem_count}")

            # Count recommendations
            with db.conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) as count FROM recommendations")
                rec_count = cur.fetchone()['count']
                print(f"recommendations row count: {rec_count}")

    except Exception as e:
        error_type = type(e).__name__
        # Sanitize error message to avoid leaking credentials
        error_msg = str(e).split(':')[0] if ':' in str(e) else str(e)
        print(f"❌ Connection failed: {error_type}: {error_msg}")


if __name__ == '__main__':
    test_connection()
