FROM node:8-buster-slim

ARG GITBOOK_VERSION=3.2.3

RUN apt-get -y update && \
    apt-get install --no-install-recommends -y curl git && \
    npm install --global gitbook-cli && \
	gitbook fetch ${GITBOOK_VERSION} && \
	npm cache clear --force && \
    rm -rf /tmp/*

WORKDIR /srv/gitbook

VOLUME /srv/gitbook /srv/html

EXPOSE 4000 35729

CMD /usr/local/bin/gitbook serve
