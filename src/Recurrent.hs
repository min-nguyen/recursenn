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
type X          = [Double]
type H          = [Double]
type Label      = [Double]
type Inputs     = ([(X, Label)], H, State)

data ForwardProp = ForwardProp {
                        fF          :: [Double],
                        iF          :: [Double],
                        oF          :: [Double],
                        aF          :: [Double],
                        prevState   :: [Double],
                        x           :: [Double],
                        h           :: [Double],
                        output      :: [Double],
                        label       :: Label
                    }

data BackProp   = BackProp {
                        deltaState_next :: [Double],
                        deltaOut_next   :: [Double],   
                        f_next          :: [Double],    
                        fpStack         :: [ForwardProp]
                    }

data Weights    = Weights {
                        fW  :: [[Double]],
                        iW  :: [[Double]],
                        aW  :: [[Double]],
                        oW  :: [[Double]]
                    }

data Biases     = Biases{
                        fB    :: [Double],
                        iB    :: [Double],
                        aB    :: [Double],
                        oB    :: [Double]
                    }   

data Cell  k = Cell {   
                        weights_w   :: Weights,
                        weights_u   :: Weights,
                        forwardProp :: ForwardProp,
                        biases      :: Biases,
                        state       :: State,
                        k           :: k
                    }
               | InputCell 


alg ::  Cell (Fix Cell, (Inputs, [ForwardProp])) -> (Fix Cell, (Inputs, [ForwardProp])) -- use forwardprop storing inputs, instead of Inputs?
alg (Cell weights_w weights_u forwardProp biases state (innerCell, ((xs, h, s), forwardPropStack))) = 
      let (x, label) = head xs
          f = map sigmoid $ eleadd3  (mvmul (fW weights_w) x) (mvmul (fW weights_u) h) (fB biases)
          i = map sigmoid $ eleadd3  (mvmul (iW weights_w) x) (mvmul (iW weights_u) h) (iB biases)
          o = map sigmoid $ eleadd3  (mvmul (oW weights_w) x) (mvmul (oW weights_u) h) (oB biases)
          a = map sigmoid $ eleadd3  (mvmul (aW weights_w) x) (mvmul (aW weights_u) h) (aB biases)
          state'  = eleadd (elemul a i) (elemul f s)
          output = elemul o (map tanh state')
          -- y = eleadd (mvmul (yW weights) output ) (yB biases)
          -- prob = softmax y
          forwardProp' = ForwardProp f i o a state' x h output label
      in  (Fx (Cell weights_w weights_u forwardProp' biases state' innerCell), 
                ((tail xs, output, state'), (forwardProp':forwardPropStack) ))

coalg :: (Fix Cell, BackProp) -> Cell (Fix Cell, BackProp)  
coalg (Fx (Cell weights_w weights_u forwardProp biases state innerCell), 
            BackProp dState_next deltaOut_next f_next fpStack)
    = let ForwardProp f i o a prevState x h output label = forwardProp
          Weights fW iW aW oW   = weights_w
          Weights fU iU aU oU   = weights_u
          deltaError = elesub output label
          dOut       = eleadd deltaError deltaOut_next
          dState     = eleadd (elemul3 dOut o (map (sub1 . sqr . tanh) state)) (elemul dState_next f_next)
          d_a        = elemul3 dState i (map (sub1 . sqr) a)
          d_i        = elemul4 dState a i (map sub1 i)
          d_f        = elemul4 dState prevState f (map sub1 f)
          d_o        = elemul4 dOut (map tanh state) o (map sub1 o)
          deltaGates = d_a ++ d_i ++ d_f ++ d_o
          weightsW   = aW  ++ iW  ++ fW  ++ oW
          weightsU   = aU  ++ iU  ++ fU  ++ oU
          deltaX     = mvmul (transpose weightsW) deltaGates
          deltaOut   = mvmul (transpose weightsU) deltaGates



      in  Cell weights_w weights_u forwardProp biases state (innerCell, 
                        BackProp dState_next deltaOut_next f_next  fpStack)