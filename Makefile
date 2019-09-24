
sync:
	rsync --info=progress2 -urltv --delete \
		--exclude 'tensorboard' --exclude 'results' --exclude 'logs' --exclude 'save' \
		-e ssh ./ dtu-data:~/workspace/stable-nalu

sync-computationally:
	rsync --info=progress2 -urltv --delete \
		--exclude 'tensorboard' --exclude 'logs' --exclude 'save' \
		-e ssh ./ computationally:~/workspace/stable-nalu

fetch-results:
	rsync --info=progress2 -urltv \
		-e ssh dtu-data:~/workspace/stable-nalu/results/ ./results

fetch-tensorboard:
	rsync --info=progress2 -urltv \
		-e ssh dtu-data:/work3/aler/tensorboard/function_task_static_nalu/ ./tensorboard/function_task_static_nalu

fetch-paper-results-from-computationally:
	rsync --info=progress2 -urltv \
		-e ssh computationally:~/workspace/stable-nalu/paper/results/ ./paper/results

clean:
	rm -rvf tensorboard/*
	rm -rvf results/*
	rm -rvf logs/*

spellcheck:
	find paper/ -name "*.tex" -exec aspell --lang=en --mode=tex --dont-backup check "{}" \;
