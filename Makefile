.PHONY: make_jupyter
make_jupyter:
	jupytext --to ipynb workshop.qmd

.PHONY: configure_buildx
configure_buildx:
	docker buildx create --name multiarch --driver docker-container --use

.PHONY: build_dev_image
build_dev_image: export TAG ?= ghcr.io/thomasjpfan/unionai-llm-tracker-workshop
build_dev_image:
	docker image build \
		--builder multiarch \
		--platform linux/arm64,linux/amd64 \
		--push \
		-f dev/Dockerfile.dev-container \
		-t ${TAG} \
		dev
