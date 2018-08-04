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
     RecordWildCards #-}

module Lib where

import Text.Show.Functions
import Data.List (transpose)


---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                        Recursive Definitions                   | | ---
  --- ‾------------------------------------------------------------------‾---

type CoAlgebra f a = a -> f a

newtype Fix f = Fx (f (Fix f))

instance (Show (f (Fix f))) => Show (Fix f) where
    showsPrec p (Fx x) = showParen (p >= 11) (showString "Fx " . showsPrec 11 x)

unFix :: Fix f -> f (Fix f)
unFix (Fx x) = x

cata :: Functor f => (f a -> a) -> Fix f -> a
cata alg = alg . fmap (cata alg) . unFix

ana :: Functor f => (a -> f a) -> (a -> Fix f)
ana coalg = Fx . fmap (ana coalg) . coalg

doggo :: Functor f => (f (Fix f, t) -> f (Fix f)) -> (f (Fix f, t) -> t) -> Fix f -> (Fix f, t)
doggo algx algy = app . fmap (doggo algx algy) . unFix
        where app = \k -> (Fx (algx k), algy k)

ella  :: Functor f =>  ((Fix f, t) -> (t -> f (Fix f, t))) -> ((Fix f, t) -> t) -> (Fix f, t) -> Fix f 
ella  algx algy = Fx . fmap (ella algx algy) . app
        where app = \k -> (algx k) (algy k )        

---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                            NN Tools                            | | ---
   --- ‾------------------------------------------------------------------‾---

sigmoid :: Double -> Double
sigmoid lx = 1.0 / (1.0 + exp (negate lx))

sigmoid' :: Double -> Double
sigmoid' x = let sig = (sigmoid x) in sig * (1.0 - sig)

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

mvmul :: Num a => [[a]] -> [a] -> [a]
mvmul mat vec = map (sum . (zipWith (*) vec)) mat

mmmul :: Fractional a => [[a]] -> [[a]] -> [[a]]
mmmul m1 m2 = [ [ sum (zipWith (*) v2 v1)  | v2 <- (transpose m2) ] |  v1 <- m1 ]

mmmul3 :: Fractional a => [[a]] -> [[a]] -> [[a]] -> [[a]]
mmmul3 m1 m2 m3 = mmmul (mmmul m1 m2) m3

elemul :: Fractional a => [a] -> [a] -> [a]
elemul v1 v2 = zipWith (*) v1 v2

elemulm :: Fractional a => [[a]] -> [[a]] -> [[a]]
elemulm m1 m2 =  [ zipWith (*) v1 v2 |  (v1, v2) <- (zip m1 m2) ]

eleaddm :: Fractional a => [[a]] -> [[a]] -> [[a]]
eleaddm m1 m2 =  [ zipWith (+) v1 v2 |  (v1, v2) <- (zip m1 m2) ]

elesubm :: Fractional a => [[a]] -> [[a]] -> [[a]]
elesubm m1 m2 =  [ zipWith (-) v1 v2 |  (v1, v2) <- (zip m1 m2) ]

fillMatrix :: Fractional a => Int -> Int -> a -> [[a]]
fillMatrix m n a = replicate m $ replicate n a

map2 :: (a -> b) -> [[a]] -> [[b]]
map2 f xs =  map (map f ) xs

map3 :: (a -> b) -> [[[a]]] -> [[[b]]]
map3 f xs = map (map (map f )) xs