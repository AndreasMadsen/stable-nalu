
sync:
	rsync --info=progress2 -urltv --delete \
		--exclude 'tensorboard' --exclude 'results' --exclude 'logs' --exclude 'save' \
		-e ssh ./ dtu-data:~/workspace/stable-nalu-copy

fetch-results:
	rsync --info=progress2 -urltv \
		-e ssh dtu-data:~/workspace/stable-nalu-copy/results/ ./results

clean:
	rm -rvf tensorboard/*
	rm -rvf results/*
	rm -rvf logs/*

spellcheck:
	find paper/ -name "*.tex" -exec aspell --lang=en --mode=tex --dont-backup check "{}" \;
