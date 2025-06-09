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


publish: build
	@echo "Publishing app" && \
	docker tag ${APP_IMAGE_NAME}:$(TAG) ${DOCKER_REGISTRY}/$(DOCKER_REPO_NAME):$(TAG) && \
	docker tag ${APP_IMAGE_NAME}:$(TAG) ${DOCKER_REGISTRY}/$(DOCKER_REPO_NAME):latest && \
	docker push ${DOCKER_REGISTRY}/${APP_IMAGE_NAME}:$(TAG) && \
	docker push ${DOCKER_REGISTRY}/${APP_IMAGE_NAME}:latest && \
	echo "Published ${DOCKER_REGISTRY}/${APP_IMAGE_NAME}:$(TAG) successfully"