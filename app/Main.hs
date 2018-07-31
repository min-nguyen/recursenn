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

module Main where

import Lib
import Data.Functor     
import Data.Foldable
import Data.Traversable
import Data.List
import Data.Ord
import Text.Show.Functions
-- import Vector
import qualified V as V
import V (Vector((:-)))
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

sigmoid :: Double -> Double
sigmoid lx = 1.0 / (1.0 + exp (negate lx))

loss :: Fractional a => [a] -> [a] -> a
loss output desired_output 
    = (1/(fromIntegral $ length output)) * (sum $ map ((\x -> x*x) . (abs)) (zipWith (-) output desired_output))


type Weights = [[Double]]
type Biases  =  [Double]
type Inputs  =  [Double]
type Activation = Double -> Double
type LossFunction = [Double] -> [Double] -> Double


data Layer' (o::V.Nat) (i::V.Nat) k where
    Layer' :: V.Vector (V.Vector Double i ) o  -> V.Vector Double i  -> Activation -> k -> Layer' o i k
    InputLayer' :: Layer' o i k 
    deriving Show
instance Functor (Layer' o i) where
    fmap eval (Layer' weights biases activate k)      = Layer' weights biases activate (eval k) 
    fmap eval (InputLayer' )                          = InputLayer' 

forward' :: Fractional a => V.Vector (V.Vector a i) o -> V.Vector a i -> (a -> a) -> ([[a]] -> [[a]]) -> ([[a]] -> [[a]]) 
forward' weights biases activate k 
    = (\inputs ->  ((map activate (zipWith (+) 
            ((V.toList $ V.map 
                ((sum)  . (zipWithPadding (*) (head inputs)) . (V.toList)) weights)) (V.toList biases))):inputs) ) . k

backward' :: (V.SingRep o, V.SingRep i, Fractional a) => V.Vector (V.Vector a i) o -> V.Vector a i -> [a] -> [a] -> V.Vector (V.Vector a i) o 
backward' weights biases input final_output 
    = let   list_weights = (V.toList $ ((V.map V.toList weights)) )
            learning_rate = 1
            desired_output = 3
            error = (sum final_output) / (fromIntegral $ length final_output)
            new_list_weights = transpose $ map (zipWith (+) (map (\xi -> learning_rate * xi * (desired_output - error)) input )) (transpose list_weights)
      in    V.unsafeFromList' $ map V.unsafeFromList' $ new_list_weights

alg' :: Layer' o i (Fix (Layer' o i), ([Inputs] -> [Inputs]) ) 
        -> (Fix (Layer' o i), ([Inputs] -> [Inputs]))
alg' (Layer' weights biases activate (innerLayer, forwardPass) )   
    =  (Fx (Layer' weights biases activate innerLayer ) , (forward' weights biases activate forwardPass) )
alg' (InputLayer' )                     
    =  (Fx InputLayer', id )

coalg' :: (V.SingRep o, V.SingRep i) => (Fix (Layer' o i), [Inputs]) -> Layer' o i (Fix (Layer' o i), [Inputs]) 
coalg' (Fx (Layer' weights biases activate innerLayer), (x:y:ys))
    =  Layer' (backward' weights biases y x) biases activate (innerLayer, (x:ys))
coalg' (Fx InputLayer', output)      
    =  InputLayer' 





data Layer k where
    Layer :: Weights -> Biases -> Activation -> k -> Layer k
    InputLayer :: Layer k 
    deriving Show

instance Functor (Layer) where
    fmap eval (Layer weights biases activate k)      = Layer weights biases activate (eval k) 
    fmap eval (InputLayer )                          = InputLayer 

forward :: Fractional a => [[a]] -> [a] -> (a -> a) -> ([[a]] -> [[a]]) -> ([[a]] -> [[a]])
forward weights biases activate k 
    = (\inputs -> (map activate ((zipWith (+) (map ((sum)  . (zipWith (*) (head inputs))) weights) biases))):inputs) . k

backward :: Fractional a => [[a]] -> [a] -> [a] -> [a] -> [[a]]
backward weights biases input final_output 
    = transpose $ map (zipWithPadding (+) (map (\xi -> learning_rate * xi * (desired_output - error)) input )) (transpose weights)
        where   learning_rate = 1
                desired_output = 3
                error = (sum final_output) / (fromIntegral $ length final_output)

alg :: Layer (Fix Layer, ([Inputs] -> [Inputs]) ) -> (Fix Layer, ([Inputs] -> [Inputs]))
alg (Layer weights biases activate (innerLayer, forwardPass) )   
    =  (Fx (Layer weights biases activate innerLayer ) , (forward weights biases activate forwardPass) )
alg (InputLayer )                     
    =  (Fx InputLayer, id)

coalg :: (Fix Layer, [Inputs]) -> Layer  (Fix Layer, [Inputs]) 
coalg (Fx (Layer weights biases activate innerLayer), (x:y:ys))
    =  Layer (backward weights biases y x) biases activate (innerLayer, (x:ys))
coalg (Fx InputLayer, output)      
    =  InputLayer 

train :: Fix Layer -> LossFunction -> Inputs -> Fix Layer 
train neuralnet lossfunction sample = ana coalg $ (nn,  diff_fun [sample])
  where 
    (nn, diff_fun) = cata alg neuralnet

trains :: Fix Layer -> LossFunction -> [Inputs] -> Fix Layer
trains neuralnet lossfunction samples  = foldr (\sample nn -> train nn lossfunction sample) neuralnet samples

construct :: [(Weights, Biases, Activation)] -> Fix Layer
construct (x:xs) = Fx (Layer weights biases activation (construct (xs)))
    where (weights, biases, activation) = x
construct []       = Fx InputLayer

zipWithPadding ::  (a -> a -> a) -> [a] -> [a] -> [a]
zipWithPadding  f (x:xs) (y:ys) = (f x y) : zipWithPadding  f xs ys
zipWithPadding  f []     ys     = ys
zipWithPadding  f xs     []     = xs


example' =  (Fx ( Layer' ((2.0:-10.0:-3.0:-V.Nil):-V.Nil) (0.0:-0.0:-0.0:-V.Nil) sigmoid
             (Fx ( Layer' ((2.0:-1.0:-3.0:-V.Nil):-V.Nil) (0.0:-0.0:-0.0:-V.Nil) sigmoid
              (Fx   InputLayer' ) ) ) ) )


example =  (Fx ( Layer [[-2.0,10.0,4.0]] [0, 0, 0] sigmoid
            (Fx ( Layer [[2.0,1.0,3.0]] [0, 0, 0] sigmoid
             (Fx   InputLayer ) ) ) ) )

main = print $ show $ train example loss [1.0, 2.0, 3.0]


