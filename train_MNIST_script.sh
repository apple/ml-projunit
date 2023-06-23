
# ------------- eps = 4 ---------------------
python train_mnist.py --epsilon 4 --epochs 10 --lr 0.1 --clip-val 1 --mechanism Gaussian --k 1  --num-rep 10

python train_mnist.py --epsilon 4 --epochs 10 --lr 0.1 --clip-val 1 --mechanism PrivUnitG --k 1  --num-rep 10

python train_mnist.py --epsilon 4 --epochs 10 --lr 0.1 --clip-val 1 --mechanism FastProjUnit --k 1000  --num-rep 10
 
python train_mnist.py --epsilon 4 --epochs 10 --lr 0.1 --clip-val 1 --mechanism FastProjUnit-corr --k 1000  --num-rep 10

python train_mnist.py --epsilon 4 --epochs 10  --lr 0.1 --clip-val 1 --mechanism RePrivHS --k 2  --num-rep 10

python train_mnist.py --epsilon 4 --epochs 10  --lr 0.1 --clip-val 1 --mechanism RePrivHS --k 1  --num-rep 10

python train_mnist.py --epsilon 4 --epochs 10  --lr 0.1 --clip-val 1 --mechanism RePrivHS --k 4  --num-rep 10



# ------------- eps = 10 ---------------------
python train_mnist.py --epsilon 10 --epochs 10 --lr 0.1 --clip-val 1 --mechanism Gaussian --k 1  --num-rep 10

python train_mnist.py --epsilon 10 --epochs 10 --lr 0.1 --clip-val 1 --mechanism PrivUnitG --k 1  --num-rep 10

python train_mnist.py --epsilon 10 --epochs 10 --lr 0.1 --clip-val 1 --mechanism FastProjUnit --k 1000  --num-rep 10

python train_mnist.py --epsilon 10 --epochs 10 --lr 0.1 --clip-val 1 --mechanism FastProjUnit-corr --k 1000  --num-rep 10

python train_mnist.py --epsilon 10 --epochs 10 --lr 0.1 --clip-val 1 --mechanism RePrivHS --k 5  --num-rep 10

python train_mnist.py --epsilon 10 --epochs 10 --lr 0.1 --clip-val 1 --mechanism RePrivHS --k 10  --num-rep 10 

python train_mnist.py --epsilon 10 --epochs 10 --lr 0.1 --clip-val 1 --mechanism RePrivHS --k 2  --num-rep 10



# ------------- eps = 16 ---------------------
python train_mnist.py --epsilon 16 --epochs 10 --lr 0.1 --clip-val 1 --mechanism Gaussian --k 1  --num-rep 10

python train_mnist.py --epsilon 16 --epochs 10 --lr 0.1 --clip-val 1 --mechanism PrivUnitG --k 1  --num-rep 10

python train_mnist.py --epsilon 16 --epochs 10 --lr 0.1 --clip-val 1 --mechanism FastProjUnit --k 1000  --num-rep 10

python train_mnist.py --epsilon 16 --epochs 10 --lr 0.1 --clip-val 1 --mechanism FastProjUnit-corr --k 1000  --num-rep 10

python train_mnist.py --epsilon 16 --epochs 10 --lr 0.1 --clip-val 1 --mechanism RePrivHS --k 8  --num-rep 10

python train_mnist.py --epsilon 16 --epochs 10 --lr 0.1 --clip-val 1 --mechanism RePrivHS --k 4  --num-rep 10

python train_mnist.py --epsilon 16 --epochs 10 --lr 0.1 --clip-val 1 --mechanism RePrivHS --k 12  --num-rep 10


