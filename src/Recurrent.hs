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
type Inputs     = ([[Double]], [[Double]])


data ForwardProp = ForwardProp {
                        fF    :: [Double],
                        iF    :: [Double],
                        oF    :: [Double],
                        cF    :: [Double]
                    }
                    
data BackProp   = BackProp {
                        dh_next     :: [Double],
                        dc_next     :: [Double],
                        nextForget  :: [Double],
                        prevState   :: [Double],
                        outputStack :: [[Double]]
                    }

data Weights    = Weights {
                        fW  :: [[Double]],
                        iW  :: [[Double]],
                        cW  :: [[Double]],
                        oW  :: [[Double]],
                        yW  :: [[Double]]
                    }

data Biases     = Biases{
                        fB    :: [Double],
                        iB    :: [Double],
                        cB    :: [Double],
                        oB    :: [Double],
                        yB    :: [Double]
                    }   

data Cell  k = Cell {   
                        weights     :: Weights,
                        forwardProp :: ForwardProp,
                        biases      :: Biases,
                        state       :: State,
                        k           :: k
                    }
               | InputCell 


alg ::  Cell (Fix Cell, Inputs) -> (Fix Cell, Inputs)
alg (Cell weights forwardProp biases state (innerCell, (xs, hs)))
    = let (x, h) = (head xs, head hs)
          hx = zipWith (\h0 x0 -> (h0:x0:[])) h x
          hf = map sigmoid $ eleadd  (mvmul (fW weights) x)  (fB biases)
          hi = map sigmoid $ eleadd  (mvmul (iW weights) x)  (iB biases)
          ho = map sigmoid $ eleadd  (mvmul (oW weights) x)  (oB biases)
          hc = map sigmoid $ eleadd  (mvmul (cW weights) x)  (cB biases)
          c  = eleadd (elemul hf state) (elemul hi hc)
          h' = elemul ho (map tanh c)
          y = eleadd (mvmul (yW weights) h' ) (yB biases)
          prob = softmax y
          forwardProp' = ForwardProp hf hi ho hc
      in (Fx (Cell weights forwardProp' biases c innerCell), ((tail xs) ++ [x], (h':hs) ))

coalg :: (Fix Cell, BackProp) -> Cell (Fix Cell, BackProp)  
coalg (Fx (Cell weights forwardProp biases state innerCell), 
            BackProp dh_next dc_next nextForget prevState outputStack)
    = let ForwardProp hf hi ho hc = forwardProp
