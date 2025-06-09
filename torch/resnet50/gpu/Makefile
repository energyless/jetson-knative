APP_IMAGE_NAME=torch-resnet50-gpu
DOCKER_REPO_NAME=torch-resnet50-gpu


build-docker-context:
	@echo "building docker context" && \
	cp -r ${CURDIR}/src ${CURDIR}/build


clear-docker-context:
	@echo "Clearing docker context" && \
	rm -rf ${CURDIR}/build/src


.PHONY: build
build: build-docker-context
	@echo "Building app" && \
	cd ${CURDIR}/build && \
	docker build --no-cache -t ${APP_IMAGE_NAME}:$(TAG) . 
	$(MAKE) clear-docker-context


