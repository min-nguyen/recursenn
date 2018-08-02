{-# LANGUAGE
     DeriveFunctor,
     DeriveFoldable,
     DeriveTraversable,
     UndecidableInstances,
     FlexibleInstances,
     ScopedTypeVariables,
     GADTs,
     DataKinds,
     KindSignatures #-}

module Lib where

---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                        Type Definitions                        | | ---
  --- ‾------------------------------------------------------------------‾---

type CoAlgebra f a = a -> f a

newtype Fix f = Fx (f (Fix f))

instance (Show (f (Fix f))) => Show (Fix f) where
    showsPrec p (Fx x) = showParen (p >= 11) (showString "Fx " . showsPrec 11 x)

data Layer k where
    Layer       :: Weights -> Biases -> (Activation, Activation') -> k -> Layer k
    InputLayer  :: Layer k 
    deriving Show

instance Functor (Layer) where
    fmap eval (Layer weights biases activate k)      = Layer weights biases activate (eval k) 
    fmap eval (InputLayer )                          = InputLayer 

data BackPropData       = BackPropData  { 
                                         inputStack     :: [Inputs], 
                                         finalOutput    :: FinalOutput, 
                                         desiredOutput  :: DesiredOutput, 
                                         outerDeltas    :: Deltas, 
                                         outerWeights   :: Weights
                                        }

type Weights            = [[Double]]
type Biases             = [Double]
type Inputs             = [Double]
type Outputs            = [Double]
type Activation         =  Double  ->  Double
type Activation'        =  Double  ->  Double
type LossFunction       = [Double] -> [Double] -> Double
type DesiredOutput      = [Double]
type FinalOutput        = [Double]
type Deltas             = [Double]

---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                        Recursive Definitions                   | | ---
  --- ‾------------------------------------------------------------------‾---

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

elemul :: Num a => [a] -> [a] -> [a]
elemul v1 v2 = zipWith (*) v1 v2