
sync:
	rsync --info=progress2 -urltv --delete \
		--exclude 'tensorboard' --exclude 'tensorboard-backup' --exclude 'results' --exclude 'logs' \
		-e ssh ./ computationally:~/workspace/stable-nalu

sync-dtu:
	rsync --info=progress2 -urltv --delete \
		--exclude 'tensorboard' --exclude 'tensorboard-backup' --exclude 'results' --exclude 'logs' \
		-e ssh ./ dtu-data:~/workspace/stable-nalu

fetch:
	rsync --info=progress2 -urltv --delete \
	-e ssh computationally:~/workspace/stable-nalu/tensorboard/ ./tensorboard
	rsync --info=progress2 -urltv --delete \
	-e ssh computationally:~/workspace/stable-nalu/results/ ./results
	rsync --info=progress2 -urltv --delete \
	-e ssh computationally:~/workspace/stable-nalu/logs/ ./logs

clean:
	rm -rvf tensorboard/*
	rm -rvf results/*
	rm -rvf logs/*
