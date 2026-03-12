FROM registry.access.redhat.com/ubi8/openjdk-21:1.20

USER root

# Create application user with uid/gid 1000
# (Pod security context enforces runAsUser: 1000)
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g 1000 -m -d /home/user appuser

# Allow arbitrary UID to add passwd/group entries at runtime (for NFS workspace mode)
RUN chmod 666 /etc/passwd /etc/group

# Copy entrypoint wrapper
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Copy Quarkus fast-jar
COPY --chown=1000:1000 target/quarkus-app/lib/ /deployments/lib/
COPY --chown=1000:1000 target/quarkus-app/quarkus/ /deployments/quarkus/
COPY --chown=1000:1000 target/quarkus-app/app/ /deployments/app/
COPY --chown=1000:1000 target/quarkus-app/quarkus-run.jar /deployments/

USER 1000

ENV HOME=/home/user

EXPOSE 8090

CMD ["/usr/local/bin/entrypoint.sh"]
