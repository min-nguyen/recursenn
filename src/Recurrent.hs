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
     ExistentialQuantification #-}

module Recurrent where

import Utils
import Data.Functor     
import Data.Foldable
import Data.Traversable
import Data.List
import Data.Ord
import Text.Show.Functions
import qualified Vector as V
import Vector (Vector((:-)))
import Debug.Trace

type State      = [Double]
type Inputs     = ([Double], [Double])

data Weights    = Weights {
                        forgetW :: [[Double]],
                        inputW  :: [[Double]],
                        outputW :: [[Double]],
                        stateW  :: [[Double]]
                    }
data Biases     = Biases{
                        forgetB :: [Double],
                        inputB  :: [Double],
                        outputB :: [Double],
                        stateB  :: [Double]
                    }
data Cell  k = Cell {   uWeights    :: Weights,
                        wWeights    :: Weights,
                        biases      :: Biases,
                        state       :: State,
                        k           :: k
                    }
               | InputCell 



alg ::  Cell (Fix Cell, Inputs) -> (Fix Cell, Inputs)
alg (Cell uWeights wWeights biases state (innerCell, (x, h)))
    = let f = map sigmoid $ eleaddv3 (mvmul (forgetW wWeights) x)   (mvmul (forgetW uWeights) h)  (forgetB biases)
          i = map sigmoid $ eleaddv3 (mvmul (inputW wWeights)  x)   (mvmul (inputW uWeights)  h)  (inputB biases)
          o = map sigmoid $ eleaddv3 (mvmul (outputW wWeights) x)   (mvmul (outputW uWeights) h)  (outputB biases)
          c = eleaddv (elemul f state)  (elemul i (map sigmoid $ eleaddv3 (mvmul (stateW wWeights) x) (mvmul (stateW uWeights) h) (stateB biases) ))
          ht = elemul o (map sigmoid c)
      in (Fx (Cell uWeights wWeights biases c innerCell), (x,ht))