{-# LANGUAGE
     DeriveFunctor,
     DeriveFoldable,
     DeriveTraversable,
     UndecidableInstances,
     FlexibleInstances,
     ScopedTypeVariables,
     MultiParamTypeClasses,
     FlexibleContexts,
     TypeFamilies,
     GADTs,
     DataKinds,
     KindSignatures,
     RecordWildCards,
     LambdaCase #-}

module Utils where
import Prelude
import Control.Lens hiding (snoc,cons)
import Text.Show.Functions
import Control.Monad    hiding (mapM, sequence)
-- import qualified Data.Functor.Fixedpoint    as F
import Data.List (transpose)
import qualified Data.Vec as V
---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                        Recursive Definitions                   | | ---
  --- ‾------------------------------------------------------------------‾---

type CoAlgebra f a = a -> f a

newtype Fix f = Fx (f (Fix f))

instance (Show (f (Fix f))) => Show (Fix f) where
    showsPrec p (Fx x) = showParen (p >= 11) (showString "Fx " . showsPrec 11 x)

unFix :: Fix f -> f (Fix f)
unFix (Fx x) = x

cataM :: (Traversable f, Monad m) => (f a -> m a) -> (Fix f -> m a)
{-# INLINE [0] cataM #-}
cataM phiM = self
    where
    self = phiM <=< (mapM self . unFix)

cata :: Functor f => (f a -> a) -> Fix f -> a
{-# INLINE [0] cata #-}
cata alg = go
    where go = alg . fmap (cata alg) . unFix

ana :: Functor f => (a -> f a) -> (a -> Fix f)
{-# INLINE [0] ana #-}
ana coalg = go
    where go = Fx . fmap (ana coalg) . coalg

meta :: Functor f => (f a -> a) -> (a -> b) -> (b -> f b) -> (Fix f -> Fix f)
meta alg e coalg = ana coalg . e . cata alg

hylo :: Functor f => (f b -> b) -> (a -> f a) -> (a -> b)
hylo phi psi = cata phi . ana psi


doggo :: Functor f => (f (Fix f, t) -> f (Fix f)) -> (f (Fix f, t) -> t) -> Fix f -> (Fix f, t)
doggo algx algy = app . fmap (doggo algx algy) . unFix
        where app = \k -> (Fx (algx k), algy k)

nana  :: Functor f =>  ((Fix f, t) -> (t -> f (Fix f, t))) -> ((Fix f, t) -> t) -> (Fix f, t) -> Fix f 
nana  algx algy = Fx . fmap (nana algx algy) . app
        where app = \k -> (algx k) (algy k )        

-- (a -> Fix g) . (g a -> a)  <=>  (F (Fix g) -> Fix g)) . (F (a -> Fix g)) Algebra Fusion (10)
algcomp :: (Functor f, Functor g) => (a -> Fix g) -> (f (Fix g) -> (Fix g)) -> (g a -> a) -> (f a -> a)
algcomp h phi phi' = (cata phi') . (phi) . (fmap h)

-- (F (g a -> Fix g)) . (a -> g a) <=> (Fix g -> F (Fix g)) . (g a -> Fix g) Algebra Fusion (10)
coalgcomp :: (Functor f, Functor g) => (Fix g -> a) -> (Fix g -> f (Fix g)) -> (a -> g a) -> (a -> f a)
coalgcomp h phi phi' = fmap h . phi . ana phi'

---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                            NN Tools                            | | ---
  --- ‾------------------------------------------------------------------‾---

sigmoid :: Double -> Double
sigmoid lx = 1.0 / (1.0 + exp (negate lx))

inverseSigmoid :: Double -> Double
inverseSigmoid lx = -log((1.0 / lx) - 1.0)

sigmoid' :: Double -> Double
sigmoid' x = let sig = (sigmoid x) in sig * (1.0 - sig)

softmax :: [Double] -> [Double]
softmax z = let denom = sum [exp (zk) | zk <- z]
            in  map (\zj -> (exp zj) / denom) z

loss :: Fractional a => [a] -> [a] -> a
loss output desired_output 
    = (1/(fromIntegral $ length output)) * (sum $ map ((\x -> x*x) . (abs)) (zipWith (-) output desired_output))

---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                         Util Functions                         | | ---
  --- ‾------------------------------------------------------------------‾---

zipWithPadding ::  (a -> a -> a) -> [a] -> [a] -> [a]
zipWithPadding  f (x:xs) (y:ys) = (f x y) : zipWithPadding  f xs ys
zipWithPadding  f []     ys     = ys
zipWithPadding  f xs     []     = xs

dot :: Fractional a => [a] -> [a] -> a
dot v1 v2 = sum $ zipWith (*) v1 v2

dotv :: Fractional a => [a] -> [a] -> [a]
dotv v1 v2 = mvmul (map (\x -> [x]) v1) v2

mvmul :: Num a => [[a]] -> [a] -> [a]
mvmul mat vec = map (sum . (zipWith (*) vec)) mat

mvmulk :: Num a => [a] -> [[a]] -> [a]
mvmulk vec mat = map (sum . (zipWith (*) vec)) mat

-- mvmulkf :: (Num a, F.FixedList f) => [a] -> f [[a]] -> f [a]
-- mvmulkf vec mat = map (sum . (zipWith (*) vec)) mat

mmmul :: Fractional a => [[a]] -> [[a]] -> [[a]]
mmmul m1 m2 = [ [ sum (zipWith (*) v2 v1)  | v2 <- (transpose m2) ] |  v1 <- m1 ]

mmmul3d :: Fractional a => [[[a]]] -> [[[a]]] -> [[[a]]]
mmmul3d m1' m2' = [[ [ sum (zipWith (*) v2 v1)  | v2 <- (transpose m2) ] |  v1 <- m1 ] | (m1, m2) <- zip m1' m2']

mmmul3 :: Fractional a => [[a]] -> [[a]] -> [[a]] -> [[a]]
mmmul3 m1 m2 m3 = mmmul (mmmul m1 m2) m3

elemul :: Fractional a => [a] -> [a] -> [a]
elemul v1 v2 = zipWith (*) v1 v2

elemul3 :: Fractional a => [a] -> [a] -> [a] -> [a]
elemul3 v1 v2 v3 = elemul v1 (elemul v2 v3)

elemul4 :: Fractional a => [a] -> [a] -> [a] -> [a]  -> [a]
elemul4 v1 v2 v3 v4 = elemul v1 (elemul3 v2 v3 v4)

elemulm :: Fractional a => [[a]] -> [[a]] -> [[a]]
elemulm m1 m2 =  [ zipWith (*) v1 v2 |  (v1, v2) <- (zip m1 m2) ]

eleadd :: Fractional a => [a] -> [a] -> [a]
eleadd v1 v2 =   zipWith (+) v1 v2 

eleaddM :: Fractional a => [[a]] -> [[a]] -> [[a]]
eleaddM m1 m2 = [ eleadd v1 v2 | (v1, v2) <- (zip m1 m2)]

eleadd3 :: Fractional a => [a] -> [a] -> [a] -> [a]
eleadd3 v1 v2 v3 =   eleadd (eleadd v1 v2) v3

eleadd3v :: (Fractional a, Num a) => V.Vec4 [a] -> V.Vec4 [a] -> V.Vec4 [a] -> V.Vec4 [a]
eleadd3v v1 v2 v3 = V.zipWith (zipWith (+)) (V.zipWith (zipWith (+)) v1 v2) v3

elesub :: Fractional a => [a] -> [a] -> [a]
elesub v1 v2 =   zipWith (-) v1 v2 

eleaddm :: Fractional a => [[a]] -> [[a]] -> [[a]]
eleaddm m1 m2 =  [ zipWith (+) v1 v2 |  (v1, v2) <- (zip m1 m2) ]

elesubm :: Fractional a => [[a]] -> [[a]] -> [[a]]
elesubm m1 m2 =  [ zipWith (-) v1 v2 |  (v1, v2) <- (zip m1 m2) ]

elesub3 :: Fractional a => [[[a]]] -> [[[a]]] -> [[[a]]]
elesub3 m1 m2 =  [ elesubm v1 v2 |  (v1, v2) <- (zip m1 m2) ]

fillMatrix :: Fractional a => Int -> Int -> a -> [[a]]
fillMatrix m n a = replicate m $ replicate n a

map2 :: (a -> b) -> [[a]] -> [[b]]
map2 f xs =  map (map f ) xs

map3 :: (a -> b) -> [[[a]]] -> [[[b]]]
map3 f xs = map (map (map f )) xs

transpose3D :: [[[a]]] -> [[[a]]]
transpose3D mat = map transpose mat

replaceElement :: [a] -> Int -> a -> [a]
replaceElement xs i x =
    fore ++ (x : aft)
    where fore = take i xs
          aft = drop (i+1) xs

sqr :: Fractional a => a -> a
sqr x = x * x

sub1 :: Fractional a => a -> a
sub1 x = 1.0 - x

cons :: a -> [a]
cons x = [x]

outerProduct :: Fractional a => [a] -> [a] -> [[a]]
outerProduct v1 v2 = mmmul (map cons v1) (cons v2)

snoc :: a -> [a] -> [a]
snoc x xs = xs ++ [x]

(!) :: V.VecList a v => v -> Int -> a
(!) v n = V.getElem (n - 1) v

class (V.VecList a v) => List a v where
    toVector :: [a] -> v

instance List Double (V.Vec4 Double) where
    toVector xs = V.fromList xs

instance List [Double] (V.Vec4 [Double]) where
    toVector xs = V.fromList xs

mapT2 :: (a -> b) -> (a, a) -> (b, b)
mapT2 f (x,y) = (f x, f y)

mapT3 :: (List a1 v1, List a2 v2) => ([a1] -> v1) -> ([a2] -> v2) -> ([a1], [a1], [a2]) -> (v1, v1, v2)
mapT3 f g (x,y,z) = (f x, f y, g z)

thrd :: (a, b, c) -> c
thrd (a, b, c) = c

tuplify2 :: [a] -> (a,a)
tuplify2 [x,y] = (x,y)

tuplify3 :: [a] -> (a,a,a)
tuplify3 [x,y,z] = (x,y,z)

tuplify4 :: [a] -> (a,a,a,a)
tuplify4 (t:x:y:z:_) = (t,x,y,z)
