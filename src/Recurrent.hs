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
import Data.List.Split
import Text.Show.Functions
import qualified Vector as V
import Vector (Vector((:-)))
import Debug.Trace

type State      = [Double]
type X          = [Double]
type H          = [Double]
type Label      = [Double]
type Inputs     = ([(X, Label)], H, State)
-- type Delta      = 
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
                        forget_next     :: [Double],
                        deltaW          :: [[Double]],
                        deltaU          :: [[Double]],
                        deltaB          :: [Double]
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

-- data Layer   =  Layer {
--                         weights_w   :: Weights,
--                         weights_u   :: Weights,
--                         biases      :: Biases
--                     }
                
data Cell  k = Cell {   
                        weights_w   :: Weights,
                        weights_u   :: Weights,
                        forwardProp :: ForwardProp,
                        biases      :: Biases,
                        state       :: State,
                        k           :: k
                    }
               | InputCell 


alg ::  Cell (Fix Cell, (Inputs)) -> (Fix Cell, (Inputs)) -- use forwardprop storing inputs, instead of Inputs?
alg (Cell weights_w weights_u forwardProp biases state (innerCell, ((xs, h, s)))) = 
      let (x, label) = head xs
          f = map sigmoid $ eleadd3  (mvmul (fW weights_w) x) (mvmul (fW weights_u) h) (fB biases)
          i = map sigmoid $ eleadd3  (mvmul (iW weights_w) x) (mvmul (iW weights_u) h) (iB biases)
          o = map sigmoid $ eleadd3  (mvmul (oW weights_w) x) (mvmul (oW weights_u) h) (oB biases)
          a = map sigmoid $ eleadd3  (mvmul (aW weights_w) x) (mvmul (aW weights_u) h) (aB biases)
          state'  = eleadd (elemul a i) (elemul f s)
          output = elemul o (map tanh state')

          forwardProp' = ForwardProp f i o a s x h output label
      in  (Fx (Cell weights_w weights_u forwardProp' biases state' innerCell), 
                (tail xs, output, state') )

coalg :: (Fix Cell, BackProp) -> Cell (Fix Cell, BackProp)  
coalg (Fx (Cell weights_w weights_u forwardProp biases state innerCell), 
            BackProp dState_next deltaOut_next f_next deltaW_next deltaU_next deltaB_next)
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
          deltaX     = mvmul (transpose weightsW) deltaGates  -- not used at the moment
          deltaOut   = mvmul (transpose weightsU) deltaGates

          deltaW     = mmmul (map cons deltaGates) (cons x) 
          deltaU     = mmmul (map cons deltaGates) (cons output) 
          deltaB     = deltaGates

          deltaW_total = (eleaddM deltaW deltaW_next)
          deltaU_total = (eleaddM deltaU deltaU_next)
          deltaB_total = (eleadd deltaB deltaB_next)
      in  Cell weights_w weights_u forwardProp biases state (innerCell, 
                        BackProp dState deltaOut f deltaW_total deltaU_total deltaB_total)


alg2 ::  Cell (Fix Cell, BackProp) -> (Fix Cell, BackProp)
alg2 (Cell weights_w weights_u forwardProp biases state (innerCell, backProp)) =
    let (deltaW_total, deltaU_total, deltaB_total) = (deltaW backProp, deltaU backProp, deltaB backProp)
        [dfW, diW, daW, doW] = deltaW_total
        [dfU, diU, daU, doU] = deltaU_total
        [dfB, diB, daB, doB] = deltaB_total
        Weights fW iW aW oW = weights_w
        Weights fU iU aU oU = weights_u
        Biases  fB iB aB oB = biases
        weightw_length      = length fW
        weightu_length      = length fU
        biases_length       = length fB
        [fW', iW', aW', oW'] = elesub3 [fW, iW, aW, oW] (map  ((map2 (0.1 *)) . (chunksOf weightw_length)) [dfW, diW, daW, doW])
        [fU', iU', aU', oU'] = elesub3 [fU, iU, aU, oU] (map  ((map2 (0.1 *)) . (chunksOf weightu_length)) [dfU, diU, daU, doU])
        [fB', iB', aB', oB'] = elesubm [fB, iB, aB, oB] (((map2 (0.1 *)) . (chunksOf biases_length)) deltaB_total)
        weights_w'  = Weights fW' iW' aW' oW'
        weights_u'  = Weights fU' iU' aU' oU'
        biases'     = Biases  fB' iB' aB' oB'
    in  (Fx (Cell weights_w' weights_u' forwardProp biases' state innerCell), backProp)


-- train :: Fix Cell -> LossFunction -> Inputs -> DesiredOutput -> Fix Cell 
-- train neuralnet lossfunction sample desiredoutput 
--     = trace (show $ head inputStack) $ 
--         ana coalg $ (nn, BackPropData inputStack desiredoutput [] [[]] )
--             where 
--                 (nn, input)      = cata alg neuralnet
