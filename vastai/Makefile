IMAGE := shahids1/vast-openpi-vega
TAG := latest

.PHONY: build build-mac run sh push pull logs exec stop clean

# Build for linux/amd64 (what Vast.ai runs)
build:
	docker buildx build --platform linux/amd64 -t $(IMAGE):$(TAG) .

# Build for native arch (fast, for testing Dockerfile logic only)
build-native:
	docker buildx build -t $(IMAGE):$(TAG) .

# Run with full Vast.ai entrypoint (mimics real instance)
run:
	docker run --rm -it --name vastai $(IMAGE):$(TAG)

# Drop straight into a shell (skip Vast.ai entrypoint)
sh:
	docker run --rm -it --name vastai --entrypoint /bin/bash $(IMAGE):$(TAG)

# Attach a second shell to an already-running container
exec:
	docker exec -it vastai /bin/bash

# Build and push in one shot
deploy: build push

push:
	docker push $(IMAGE):$(TAG)

pull:
	docker pull $(IMAGE):$(TAG)

# Show what's running
logs:
	docker logs -f vastai

stop:
	docker stop vastai

# Nuclear options
clean:
	docker builder prune -f
	docker image rm $(IMAGE):$(TAG) 2>/dev/null || true
