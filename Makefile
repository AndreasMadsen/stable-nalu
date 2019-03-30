
sync:
	rsync --info=progress2 -urltv --delete \
		--exclude 'tensorboard' --exclude 'results' --exclude 'logs' \
		-e ssh ./ computationally:~/workspace/stable-nalu

fetch:
	rsync --info=progress2 -urltv --delete \
	-e ssh computationally:~/workspace/stable-nalu/runs/ ./tensorboard
	rsync --info=progress2 -urltv --delete \
	-e ssh computationally:~/workspace/stable-nalu/results/ ./results
	rsync --info=progress2 -urltv --delete \
	-e ssh computationally:~/workspace/stable-nalu/logs/ ./logs

clean:
	rm -rvf runs/*
