{-# LANGUAGE
     DeriveFunctor,
     DeriveFoldable,
     DeriveTraversable,
     TemplateHaskell, RankNTypes, DeriveFoldable,
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
import Prelude
import Utils
import Data.Functor     
import Data.Foldable
import Data.Maybe
import Data.Traversable
import Control.Lens hiding (snoc,cons)
import Data.List
import qualified Data.Vec as V
import Data.Ord
import Data.List.Split
import Text.Show.Functions
import Debug.Trace

type State      = [Double]
type X          = [Double]
type Label      = [Double]
type Inputs     = [(X, Label)]

--f i a o

type Gates      =  V.Vec4 [Double] 
type Weights    =  V.Vec4 [[Double]] 
type Biases     =  V.Vec4 [Double] 

type HyperParameters = (Weights, Weights, Biases)

data ForwardProp = ForwardProp {
                        _gates       :: Gates,
                        _x           :: [Double],
                        _h           :: [Double],
                        _label       :: Label,
                        _output      :: [Double],
                        _prevState   :: [Double],
                        _params      :: HyperParameters,
                        _inputStack  :: Inputs
                    } deriving Show
makeLenses ''ForwardProp

data BackProp   = BackProp {
                        _nextDState  :: [Double],
                        _nextDOut    :: [Double], 
                        _nextDGates  :: [Double],  
                        _nextF      :: [Double],
                        _nextLayerDeltas  :: [[Double]],
                        _nextLayerWeights :: Weights
                    } deriving Show
makeLenses ''BackProp


data Deltas  = Deltas {
                        deltaW           :: [[Double]],
                        deltaU           :: [[Double]],
                        deltaB           :: [Double]
                }
                | NoDeltas deriving Show

data Cell  k =   Cell {   
                        cellState   :: State,
                        cellDeltas  :: Deltas,
                        innerCell   :: k
                     }
                | EndCell {
                        cellState   :: State,
                        cellDeltas  :: Deltas,
                        innerCell   :: k
                     }
                | InputCell  deriving (Functor, Show)
makeLenses ''Cell
data Layer k =  Layer {
                        _hparams     :: HyperParameters,
                        _cells       :: Fix Cell,
                        _innerLayer  :: k
                    }
                | InputLayer deriving (Functor, Show)
makeLenses ''Layer

runs :: Layer k -> Layer k 
runs InputLayer = InputLayer
runs (Layer params cells innerLayer)
    = let (w,u,b) = params
          initialForwardProp = ForwardProp (V.fromList [[],[],[],[]]) [] [] [] [0] [0] params [([1,2],[0.5]),([0.5,3], [1.25])]
          initialBackProp    = BackProp [0] [0] [0,0,0,0] [0] [[]] (V.fromList [[],[],[],[]])

          (cellf, deltaTotalFunc) = ((cata alg2_cell) . (ana coalg_cell) . (\(c, f) -> (c, f [initialForwardProp], initialBackProp)) . (cata alg_cell)) cells 

          h                         = length $ w ! 1
          deltaTotal                = deltaTotalFunc $ Deltas (fillMatrix (4 * h) (length $ fst $ head $ initialForwardProp ^. inputStack) 0.0) 
                                                              (fillMatrix (4 * h) (h) 0.0)
                                                              (replicate  (4 * h) 0.0)

          layer'                     = updateParameters (Layer params cellf innerLayer) deltaTotal
      in  layer'


compGates :: HyperParameters -> [Double] -> [Double] -> Gates
compGates (weightsW, weightsU, biases) x h 
    =  V.map ((snoc tanh (replicate 3 sigmoid)) <*>) (eleadd3v (V.map (mvmulk x) weightsW) (V.map (mvmulk h) weightsU) biases)


alg_cell ::  Cell (Fix Cell, [ForwardProp] -> [ForwardProp]) -> (Fix Cell, [ForwardProp] -> [ForwardProp]) -- use forwardprop storing inputs, instead of Inputs?
alg_cell InputCell = 
    (Fx InputCell, id)
alg_cell cell
    = let (state, deltas, (nextCell, forwardProps)) = (cellState cell, cellDeltas cell, innerCell cell)
          forwardProps' = (\fps -> 
                let fp = head fps
 
                    (x, label) = head (fp^.inputStack)
                    gates   = compGates (fp^.params) x (fp^.output)
                    state'  = eleadd (elemul (gates ! 3) (gates ! 2)) (elemul (gates ! 1) (fp^.prevState))
                    output' = elemul (gates ! 4) (map tanh state')
                in  ((ForwardProp gates x  (fp^.output) label output' state'  (fp^.params) (tail (fp^.inputStack))):fps)) . forwardProps
      in  (Fx (cell {innerCell = nextCell}), forwardProps')

coalg_cell :: (Fix Cell, [ForwardProp], BackProp) -> Cell (Fix Cell, [ForwardProp], BackProp) 
coalg_cell (Fx (EndCell state deltas innerCell), forwardProps, backProp)
  = let fp = head forwardProps
        lastState               =   (head (tail forwardProps)) ^. prevState 
        (gate, updatedState)    = (fp ^. gates, fp ^. prevState)
        (weightsW, weightsU)    = mapT2 (V.foldr (++) [[]]) (fp^.params._1, fp^.params._2)
    
--      nextLayerDelta = head nextLayerDeltas
--      deltaError = elemul (mvmul (transpose nextLayerWeightsW) nextLayerDelta) x

        deltaError = elesub (fp^.output) (fp^.label)
        dOut       = deltaError
        dState     = (elemul3 dOut (gate ! 4) (map (sub1 . sqr . tanh) updatedState))

        d_f        = elemul4 dState lastState (gate ! 1) (map sub1 (gate ! 1))
        d_i        = elemul4 dState (gate ! 3) (gate ! 2) (map sub1 (gate ! 2))
        d_a        = elemul3 dState (gate ! 2) (map (sub1 . sqr) (gate ! 3))
        d_o        = elemul4 dOut (map tanh updatedState) (gate ! 4) (map sub1 (gate ! 4))
        
        deltaGates = d_f ++ d_i ++ d_a ++ d_o
        (deltaX, deltaOut)     = mapT2 (mvmulk deltaGates . transpose) (weightsW, weightsU)  -- not used at the moment
      
        deltaW     = outerProduct deltaGates (fp^.x) 
        deltaU     = fillMatrix (4 * (length ((fp^.params._2) ! 1))) (length $ head ((fp^.params._2) ! 1)) (0.0 :: Double)
        deltaB     = deltaGates
        backProp'  = backProp & nextDState .~ dState 
                              & nextDOut   .~ deltaOut 
                              & nextDGates .~ deltaGates 
                              & nextF      .~ (gate ! 1) 

    in EndCell updatedState (Deltas deltaW deltaU deltaB) (innerCell, tail forwardProps, backProp' )

coalg_cell (Fx (Cell state deltas innerCell), forwardProps, backProp)
  = let fp = head forwardProps
        lastState               = (head (tail forwardProps)) ^. prevState 
        (gate, updatedState)    = (fp ^. gates, fp ^. prevState)
        (weightsW, weightsU)    = mapT2 (V.foldr (++) [[]]) (fp^.params._1, fp^.params._2)
    
        BackProp dState_next deltaOut_next deltaGates_next
                    f_next nextLayerDeltas nextLayerWeightsW = backProp

        deltaError = elesub (fp^.output)  (fp^.label)
        dOut       = eleadd deltaError deltaOut_next
        dState     = eleadd (elemul3 dOut (gate ! 4) (map (sub1 . sqr . tanh) updatedState)) (elemul dState_next f_next)
        
        d_f        = elemul4 dState lastState (gate ! 1) (map sub1 (gate ! 1))
        d_i        = elemul4 dState (gate ! 3) (gate ! 2) (map sub1 (gate ! 2))
        d_a        = elemul3 dState (gate ! 2) (map (sub1 . sqr) (gate ! 3))
        d_o        = elemul4 dOut (map tanh updatedState) (gate ! 4) (map sub1 (gate ! 4))
        
        deltaGates = d_f ++ d_i ++ d_a ++ d_o
        (deltaX, deltaOut)     = mapT2 (mvmulk deltaGates . transpose) (weightsW, weightsU)

        deltaW     = outerProduct deltaGates (fp^.x) 
        deltaU     = outerProduct deltaGates_next (fp^.output)
        deltaB     = deltaGates
        backProp'  = BackProp dState deltaOut deltaGates (gate ! 1)  nextLayerDeltas nextLayerWeightsW

      in Cell updatedState (Deltas deltaW deltaU deltaB) 
                        (innerCell, tail forwardProps, backProp' )
coalg_cell (Fx InputCell, forwardProps, backProp)
    = InputCell


alg2_cell ::  Cell (Fix Cell, Deltas -> Deltas) ->  (Fix Cell, Deltas -> Deltas)
alg2_cell InputCell 
    = (Fx InputCell, id)

alg2_cell cell
    =   let (state, deltas, (nextCell, deltaTotalFunc)) = (cellState cell, cellDeltas cell, innerCell cell)
            Deltas deltaW1 deltaU1 deltaB1 = deltas
            deltaTotalFunc' = (\deltaTotal -> 
                let Deltas deltaW2 deltaU2 deltaB2 = deltaTotal

                    deltaW_total = (eleaddM deltaW1 deltaW2)
                    deltaU_total = (eleaddM deltaU1 deltaU2) -- verified
                    deltaB_total = (eleadd deltaB1 deltaB2)
                in  Deltas deltaW_total deltaU_total deltaB_total) . deltaTotalFunc
        in  (Fx (cell {innerCell = nextCell}), deltaTotalFunc') --


updateParameters ::  Layer k -> Deltas -> Layer k
updateParameters layer delta_total
    =   let Deltas deltaW_total deltaU_total deltaB_total = delta_total
            (weights_w,weights_u,biases) = fromJust $ layer ^? hparams
            [fW, iW, aW, oW] = V.toList weights_w
            [fU, iU, aU, oU] = V.toList weights_u
            [fB, iB, aB, oB ]= V.toList biases
            w     = V.fromList $ map cons $ elesubm (fW ++ iW ++ aW ++ oW) (map2 (0.1 *) deltaW_total)
            u     = V.fromList $ map cons $ elesubm (fU ++ iU ++ aU ++ oU) (map2 (0.1 *) deltaU_total)
            b     = V.fromList $ map cons $ elesub  (fB ++ iB ++ aB ++ oB) (map (0.1 *) deltaB_total)

        in layer & hparams .~ (w,u,b)


-- updateParameters ::  Layer k -> Deltas -> Layer k
-- updateParameters layer delta_total
--     =   let Deltas deltaW_total deltaU_total deltaB_total = delta_total

--             (weightsW, weightsU) = mapT2 (V.foldl (++) [[]]) (layer^.hparams._1, layer^.hparams._2)
--             biases               = (V.foldl (++) [] (layer^.hparams._3))
--             w'                   =  elesubm (weightsW) (map2 (0.1 *) deltaW_total)
--             u'                   =  elesubm (weightsU) (map2 (0.1 *) deltaU_total)
--             b'                   =  elesub  (biases) (map  (0.1 *) deltaB_total)
--             f                    =  V.fromTuple . tuplify4 . map cons
--             hparams'             =  (f w', f u', f b')

--         in layer & hparams .~ hparams'

example = Layer (V.fromList [[[0.7, 0.45]],  [[0.95, 0.8]],  [[0.45, 0.25]],   [[0.6, 0.4]]],
                 V.fromList [[[0.1]]      ,  [[0.8]]      ,   [[0.15]]     ,    [[0.25]]],
                 V.fromList [[0.15]       , [0.65]         , [0.2]        ,    [0.1]])
                (Fx (EndCell [0.68381] NoDeltas (Fx (Cell [0] NoDeltas (Fx InputCell))))) (Fx InputLayer)

runRecurrent =  print $ show $ runs example