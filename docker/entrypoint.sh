#!/bin/sh
# yantrikdb container entrypoint.
#
# Solves issue #35 part 2: when users mount a host directory at
# /var/lib/yantrikdb (e.g. `-v ./data:/var/lib/yantrikdb`), the host
# directory is typically owned by their host UID (often 1000), while
# the in-container `yantrikdb` user is a system account with a different
# UID assigned by `useradd -r`. SQLite then fails to open the DB:
#
#     Error: unable to open database file: /var/lib/yantrikdb/control.db
#
# This entrypoint ensures the mounted data directory is owned by the
# container's `yantrikdb` user BEFORE dropping privileges. Standard
# Postgres/Redis Docker convention.
#
# The container must start as root for the `chown` to work; we
# explicitly drop to the unprivileged `yantrikdb` user via `gosu`
# before exec'ing the actual server command.

set -e

DATA_DIR="${YANTRIKDB_DATA_DIR:-/var/lib/yantrikdb}"
RUN_USER="${YANTRIKDB_RUN_USER:-yantrikdb}"

# If running as root, ensure DATA_DIR is owned by the run user, then
# drop privileges. If already running as a non-root user (e.g. via
# `docker run -u 1000:1000`), skip the chown — the user took responsibility
# for ownership themselves.
if [ "$(id -u)" = "0" ]; then
    # Create DATA_DIR if missing (e.g. fresh anonymous volume).
    mkdir -p "$DATA_DIR"

    # Chown only if not already correct, to avoid unnecessary work on
    # large data directories. `-R` because /etc/yantrikdb may live
    # inside on Bind-mounted configs too.
    CURRENT_OWNER=$(stat -c '%U' "$DATA_DIR" 2>/dev/null || echo "")
    if [ "$CURRENT_OWNER" != "$RUN_USER" ]; then
        echo "[entrypoint] chowning $DATA_DIR to $RUN_USER (was: $CURRENT_OWNER)" >&2
        chown -R "$RUN_USER:$RUN_USER" "$DATA_DIR"
    fi

    # Drop to unprivileged user via gosu.
    exec gosu "$RUN_USER" "$@"
fi

# Already non-root: exec directly.
exec "$@"
