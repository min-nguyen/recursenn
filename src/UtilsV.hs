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

module UtilsV where

import Text.Show.Functions
import Data.List (transpose)
import qualified Data.Vector.Sized as V



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

inverseSigmoid :: Double -> Double
inverseSigmoid lx = -log((1.0 / lx) - 1.0)

sigmoid' :: Double -> Double
sigmoid' x = let sig = (sigmoid x) in sig * (1.0 - sig)

loss :: Fractional a => [a] -> [a] -> a
loss output desired_output 
    = (1/(fromIntegral $ length output)) * (sum $ map ((\x -> x*x) . (abs)) (zipWith (-) output desired_output))


dot :: Fractional a => V.Vector n a -> V.Vector n a -> a
dot v1 v2 = V.sum $ V.zipWith (*) v1 v2


eleadd :: Fractional a => V.Vector n a -> V.Vector n a -> V.Vector n a 
eleadd v1 v2 =   V.zipWith (+) v1 v2 


eleadd3 :: Fractional a => V.Vector n a -> V.Vector n a -> V.Vector n a -> V.Vector n a
eleadd3 v1 v2 v3 =  eleadd (eleadd v1 v2) v3


elemul ::  Fractional a => V.Vector n a -> V.Vector n a -> V.Vector n a
elemul v1 v2 = V.zipWith (*) v1 v2

elemul3 ::  Fractional a => V.Vector n a -> V.Vector n a -> V.Vector n a -> V.Vector n a
elemul3 v1 v2 v3 = elemul v1 (elemul v2 v3)

elemul4 ::  Fractional a => V.Vector n a -> V.Vector n a -> V.Vector n a -> V.Vector n a -> V.Vector n a
elemul4 v1 v2 v3 v4 = elemul v1 (elemul3 v2 v3 v4)

mvmul ::Fractional a =>  V.Vector m (V.Vector n a) -> V.Vector n a ->  V.Vector m a
mvmul mat vec = V.map (V.sum . (V.zipWith (*) vec)) mat
