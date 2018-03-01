CC=/usr/local/cuda/bin/nvcc
INCLUDE=-I/usr/local/cuda/include \

SOURCE=driver.cu
EXECUTABLE=sobeledgeSharedUnroll
FLAGS=
#-g -G
#FOR DEBUGGING
#FLAGS=-g

$(EXECUTABLE): $(SOURCE)
	$(CC) $(FLAGS) $(INCLUDE) $< -o $@ 

clean:
