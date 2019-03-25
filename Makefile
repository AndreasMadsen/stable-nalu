
sync:
	rsync --info=progress2 -urltv --delete \
		--exclude 'runs' --exclude 'results' \
		-e ssh ./ computationally:~/workspace/stable-nalu

fetch:
	rsync --info=progress2 -urltv --delete \
	-e ssh computationally:~/workspace/stable-nalu/runs/ ./runs
	rsync --info=progress2 -urltv --delete \
	-e ssh computationally:~/workspace/stable-nalu/results/ ./results

clean:
	rm -rvf runs/*
