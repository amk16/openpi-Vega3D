IMAGE := shahids1/vast-openpi-vega
TAG := latest
LIBERO_DATA_DIR := libero-data

.PHONY: build build-mac build-native run sh push pull logs exec stop clean sync-libero-data

# Sync LIBERO bddl_files and init_files from S3 into the build context.
# Done on the host (not in Dockerfile) because the Dockerfile has no AWS auth.
sync-libero-data:
	mkdir -p $(LIBERO_DATA_DIR)/bddl_files $(LIBERO_DATA_DIR)/init_files
	aws s3 sync s3://behavior-challenge/libero_bddl_files/ $(LIBERO_DATA_DIR)/bddl_files/
	aws s3 sync s3://behavior-challenge/libero_init_files/ $(LIBERO_DATA_DIR)/init_files/

# Build for linux/amd64 (what Vast.ai runs)
build: sync-libero-data
	docker buildx build --platform linux/amd64 -t $(IMAGE):$(TAG) -f Dockerfile ..

# Build for native arch (fast, for testing Dockerfile logic only)
build-native: sync-libero-data
	docker buildx build -t $(IMAGE):$(TAG) -f Dockerfile ..

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
