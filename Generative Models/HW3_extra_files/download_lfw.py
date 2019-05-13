import requests
import sys, os
import tarfile

def download_lfw_dataset(url, filename):
	
	data_directory = 'data'
	
	#create data directory if not exist
	if not os.path.exists(data_directory):
		os.makedirs(data_directory)
	
	#check if file already downloaded
	if os.path.exists(filename):
		print ('Dataset was downloaded before, exit.')
		return 
	
	#download file
	with open(filename, "wb") as f:
			print ("Downloading %s" % filename)
			response = requests.get(url, stream=True)
			total_length = response.headers.get('content-length')

			if total_length is None: # no content length header
				f.write(response.content)
			else:
				dl = 0
				total_length = int(total_length)
				for data in response.iter_content(chunk_size=total_length//100):
					dl += len(data)
					f.write(data)
					done = int(50 * dl / total_length)
					sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
					sys.stdout.flush()
					
	#extruct .tar file
	with tarfile.open(filename, 'r') as f:
		f.extractall(data_directory)


if __name__ == "__main__":
    url = 'http://www.briancbecker.com/blog/research/pubfig83-lfw-dataset/'
    filename = 'lfw_data'
    download_lfw_dataset(url, filename)