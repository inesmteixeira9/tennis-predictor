.DEFAULT_GOAL := help

PROJECTNAME := $(shell basename $(PWD))
BRANCH      := $(shell git rev-parse --abbrev-ref HEAD)
VERSION     := $(shell git describe --tags --abbrev=0)

ifeq ($(BRANCH),master)
	ENV := prod
	IMG_TAG := $(VERSION)
else
	ENV := dev
	IMG_TAG := $(VERSION)-$(ENV)
endif

ifeq (test-db, $(firstword $(MAKECMDGOALS)))
  runargs := $(wordlist 2, $(words $(MAKECMDGOALS)), $(MAKECMDGOALS))
  $(eval $(runargs):;@true)
endif


-include env/env.$(ENV)
export

APP_NAME := $(PROJECTNAME)
IMG_APP  := $(APP_REPO)/$(APP_NAME)

BUILD_ARGS := $(shell echo "--build-arg BRANCH=$(BRANCH) --build-arg http_proxy=$(http_proxy) --build-arg https_proxy=$(http_proxy) --build-arg noproxy=$(no_proxy) --build-arg REGISTRY=$(REGISTRY) --build-arg BASE_IMG=$(BASE_IMG) --build-arg APP_NAME=$(APP_NAME) --build-arg APP_PORT=$(APP_PORT)")
COMPOSE_ARGS := $(shell echo "ENV=$(ENV) APP_NAME=$(APP_NAME) APP_DEBUG=$(APP_DEBUG) REGISTRY=$(REGISTRY) IMG_APP=$(IMG_APP) IMG_TAG=$(IMG_TAG) APP_PORT=$(APP_PORT) HOST_PORT=$(HOST_PORT) APP_URL=$(APP_URL) http_proxy=$(http_proxy) https_proxy=$(http_proxy) noproxy=$(no_proxy) GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/$(APP_NAME)-sa.json")

.PHONY: env help build push deploy rm ps ls-images rm-images bash test

objects = rm build rm-config create-config push deploy
all: $(objects)

env:
	@echo "Environment is selected based on git branch"
	@echo " (master -> prod | staging -> dev)"
	@echo "-------------------------------------------------------"
	@echo "Git project         : $(PROJECTNAME)"
	@echo "Git branch          : $(BRANCH)"
	@echo "Git version         : $(VERSION)"
	@echo "APP Env             : $(ENV)"
	@echo "APP Debug           : $(APP_DEBUG)"
	@echo "APP Image Name      : $(IMG_APP):$(IMG_TAG)"
	@echo "APP Base Image      : $(BASE_IMG)"
	@echo "Docker Registry     : $(REGISTRY)"
	@echo "Docker build args   : $(BUILD_ARGS)"
	@echo "Docker compose args : $(COMPOSE_ARGS)"
	@echo "-------------------------------------------------------"
	@echo ""
help: env
	@echo "Usage: make command"
	@echo ""
	@echo "Commands:"
	@echo "  git-init      Git init in current directory and setup remote url"
	@echo "  build         Build images using Dockerfile to local repository"
	@echo "  push          Push images from local to remote repository"
	@echo "  deploy        Deploy stack on docker swarm"
	@echo "  ps            List running containers from stack"
	@echo "  ls-images     List images with filter='nos-aid/nospra*'"
	@echo "  rm            Remove stack from docker swarm"
	@echo "  rm-images     Delete images from local and remote repository"
	@echo "  logs          Show $(APP_NAME)_app service logs --follow"
	@echo "  bash          Open bash on $(APP_NAME)_app service"
	@echo "  stats         Show container(s) resource usage statistics"
	@echo "  top           Display the running processes of a container"
	@echo "  health        Check health of service"
	@echo ""

build:
	@echo ">>>>> Build image $(IMG_APP):$(IMG_TAG)"
	@docker build --no-cache $(BUILD_ARGS) -t $(IMG_APP):$(IMG_TAG) -f env/Dockerfile .
	@echo "" 
push:
	@echo ">>>>> Push image $(IMG_APP):$(IMG_TAG) to $(REGISTRY)/$(IMG_APP):$(IMG_TAG)"
	docker tag $(IMG_APP):$(IMG_TAG) $(REGISTRY)/$(IMG_APP):$(IMG_TAG)
	docker push $(REGISTRY)/$(IMG_APP):$(IMG_TAG)
	@echo "" 
deploy:
	@echo ">>>>> Deploy stack $(APP_NAME)"
	@$(COMPOSE_ARGS) docker stack deploy -c env/docker-compose.$(ENV) $(APP_NAME)
	@echo ""
	@echo ">>>> Check health on https://$(APP_URL)/$(APP_NAME)/api/v1/health"
	@echo ">>>> Check API docs on https://$(APP_URL)/$(APP_NAME)/api/v1/docs"
	@echo ">>>> Check health running 'make health' on docker swarm server" 
	@echo ">>>> Check health running 'make health' inside container" 

ps:
	docker stack ps $(APP_NAME) --no-trunc
ls-images:
	docker images | grep $(PROJECTNAME)
rm:
	@echo ">>>>> Remove stack $(APP_NAME)"
	docker stack rm $(APP_NAME)
rm-images:
	@echo ">>>>> Delete local image $(IMG_APP):$(IMG_TAG) and remote $(REGISTRY)/$(IMG_APP):$(IMG_TAG)"
	@if [[ -n $$(docker images -qa "$(IMG_APP):$(IMG_TAG)") ]]; then docker rmi -f $(IMG_APP):$(IMG_TAG); fi	
	@if [[ -n $$(docker images -qa "$(REGISTRY)/$(IMG_APP):$(IMG_TAG)") ]]; then docker rmi -f $(REGISTRY)/$(IMG_APP):$(IMG_TAG); fi
	docker rmi `docker images --filter dangling=true -q`	
logs:
	docker service logs $(APP_NAME)_app --follow --raw
bash:
	docker exec -it $$(docker ps -f name=$(APP_NAME)_app -f status=running -q) bash
stats:
	docker stats $$(docker ps --filter name=$(APP_NAME)_app -q)
top:
	docker top $$(docker ps --filter name=$(APP_NAME)_app -q)

create-config:
	docker config create $(APP_NAME)-config.toml config/$(APP_NAME)-config.toml
rm-config:
	docker config rm $(APP_NAME)-config.toml
create-docs:
	docker config create $(APP_NAME)-docs.toml docs/$(APP_NAME)-docs.toml
rm-docs:
	docker config rm $(APP_NAME)-docs.toml

create-secret:
	docker secret create $(APP_NAME)-sa.json $(APP_NAME)-sa.json
rm-secret:
	docker secret rm $(APP_NAME)-sa.json

health:
	@curl -s -L -k http://$(APP_URL)/$(APP_NAME)/api/v1/health

git-init:
	rm -rf .git
	mv config/app-baseline-config.toml config/$(APP_NAME)-config.toml
	sed -i 's/app-baseline-config/$(APP_NAME)-config/g' config/$(APP_NAME)-config.toml
	sed -i 's/app-baseline-config/$(APP_NAME)-config/g' env/docker-compose.prod
	sed -i 's/app-baseline-config/$(APP_NAME)-config/g' env/docker-compose.dev
	sed -i 's/app-baseline-sa/$(APP_NAME)-sa/g' env/docker-compose.prod
	sed -i 's/app-baseline-sa/$(APP_NAME)-sa/g' env/docker-compose.dev
	sed -i 's/app-baseline-config/$(APP_NAME)-config/g' app/config.py
	git init --initial-branch=master 
	@read -p "Please enter remote url: " REMOTE_URL; \
	git remote add origin $$REMOTE_URL
	git add .
	git commit -m "Initial commit"
	git pull origin master
	git push -u origin master
	git tag -a 0.1.0 -m "version 0.1.0"
	git push --tags
	git checkout -b dev
	git pull origin dev
	git push -u origin dev
	
lint:
	blue --diff .
	isort --profile black -l 79 --diff .

lint-check:
	blue --check --diff .
	isort --profile black -l 79 --check --diff .

test:
	docker exec -it $$(docker ps -f name=$(APP_NAME)_app -f status=running -q) pytest
test-api:
	docker exec -it $$(docker ps -f name=$(APP_NAME)_app -f status=running -q) pytest -m api -rA
test-onprem:
	docker exec -it $$(docker ps -f name=$(APP_NAME)_app -f status=running -q) pytest -m db_onprem -rA
test-bq:
	docker exec -it $$(docker ps -f name=$(APP_NAME)_app -f status=running -q) pytest -m db_bq
test-aid:
	docker exec -it $$(docker ps -f name=$(APP_NAME)_app -f status=running -q) pytest -m aid_libs
test-debug:
	docker exec -it $$(docker ps -f name=$(APP_NAME)_app -f status=running -q) pytest -rA
