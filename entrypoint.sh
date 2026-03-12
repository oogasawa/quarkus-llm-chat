#!/bin/bash
# Wrapper entrypoint: add /etc/passwd entry for arbitrary UID, then start the app.

CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)
if ! id -un "$CURRENT_UID" >/dev/null 2>&1; then
    echo "appuser:x:${CURRENT_UID}:${CURRENT_GID}:App User:${HOME}:/bin/bash" >> /etc/passwd
    echo "Added passwd entry for UID ${CURRENT_UID}"
fi
if ! getent group "$CURRENT_GID" >/dev/null 2>&1; then
    echo "appuser:x:${CURRENT_GID}:" >> /etc/group
    echo "Added group entry for GID ${CURRENT_GID}"
fi

exec java -jar /deployments/quarkus-run.jar
