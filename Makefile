.PHONY: test local-deploy remote-deploy pipeline

test:
	./gradlew testDebugUnitTest

local-deploy: test
	./gradlew installDebug

build-release:
	./gradlew assembleRelease

commit-push:
	git add .
	@read -p "Commit message: " msg; \
	if [ -z "$$msg" ]; then \
		git commit -m "Auto-commit from Makefile"; \
	else \
		git commit -m "$$msg"; \
	fi
	git push origin main

pipeline: test local-deploy commit-push
	@echo "Pipeline finished!"
