.PHONY: push

push:
	git add .
	git commit -m "$(commit)"
	git push