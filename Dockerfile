FROM registry.access.redhat.com/ubi8/openjdk-21:1.20

# Copy Quarkus fast-jar
COPY --chown=185:185 target/quarkus-app/lib/ /deployments/lib/
COPY --chown=185:185 target/quarkus-app/quarkus/ /deployments/quarkus/
COPY --chown=185:185 target/quarkus-app/app/ /deployments/app/
COPY --chown=185:185 target/quarkus-app/quarkus-run.jar /deployments/

USER 185

EXPOSE 8090

CMD ["java", "-jar", "/deployments/quarkus-run.jar"]
