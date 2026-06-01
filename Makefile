CC=gcc

ifeq ($(OS),Windows_NT)
LIB=conv2d.dll
LIBFLAGS=-shared -O2
EXE=infer.exe
else
LIB=libconv2d.so
LIBFLAGS=-shared -O2 -fPIC
EXE=infer
endif

lib:
	$(CC) $(LIBFLAGS) conv2d.c -o $(LIB)

train: lib
	python train.py

export:
	python export_weights.py

infer:
	$(CC) infer.c conv2d.c -o $(EXE) -lm

clean:
	rm -f libconv2d.so conv2d.dll infer infer.exe model_weights.pth weights.bin samples.bin training_log.txt python_predictions.txt inference_log.txt accuracy_plot.png loss_plot.png
