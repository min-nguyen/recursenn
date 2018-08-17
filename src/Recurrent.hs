
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
import Control.Applicative
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
                        _nextDState         :: [Double],
                        _nextDOut           :: [Double], 
                        _nextDGates         :: [Double],  
                        _nextF              :: [Double],
                        _nextLayerDXs       :: Maybe [[Double]],
                        _nextLayerWs        :: Maybe Weights
                    } deriving Show
makeLenses ''BackProp


data Deltas  = Deltas {
                        deltaW           :: [[Double]],
                        deltaU           :: [[Double]],
                        deltaB           :: [Double],
                        deltaXs          :: [[Double]]
                }
                | NoDeltas deriving Show

data Cell  k =   Cell {   
                        _cellState   :: State,
                        _cellDeltas  :: Deltas,
                        _innerCell   :: k
                     }
                | EndCell {
                        _cellState   :: State,
                        _cellDeltas  :: Deltas,
                        _innerCell   :: k
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

runs' :: Fix Layer  -> Fix Layer
runs' layer        = let (layer', forwardProp') = cata alg_layer layer
                         sample = [([1,2],[0.5]),([0.5,3], [1.25])]
                     in ana coalg_layer (layer', forwardProp' sample, (initBackProp 1 2))

alg_layer :: Layer (Fix Layer, Inputs -> [[ForwardProp]]) -> (Fix Layer, Inputs -> [[ForwardProp]])
alg_layer InputLayer = (Fx InputLayer, (\sample -> cons [ForwardProp emptyGates [0] [0] l x [0] emptyParams sample | (x,l) <- sample  ]))
alg_layer (Layer params cells (innerLayer, nextForwardProp))
    = let forwardProp  = (\(fps :: [[ForwardProp]]) ->
                    let fp = head fps
                        inputs = map tuplify2 $ chunksOf 2 $ fp <**> [(^.output), (^.label)] 
                        (hDim, dDim)       = let (w,u,b) = params in (length $ w ! 1, length $ head $ w ! 1)
                        initialForwardProp = initForwardProp hDim dDim params inputs
                        (cell, fpFunc)     = cata alg_cell cells 
                        layerFP            = fpFunc [initialForwardProp]
                    in  (layerFP:fps)) . nextForwardProp
      in (Fx (Layer params cells innerLayer), forwardProp) 

coalg_layer :: (Fix Layer, [[ForwardProp]], BackProp) -> Layer (Fix Layer, [[ForwardProp]], BackProp)
coalg_layer (Fx InputLayer, fp, bp)
            = InputLayer
coalg_layer (Fx (Layer params cells innerLayer), fps, backProp)
    =   let fp                  = head fps
            (w,u,b)             = params
            (hDim, dDim)        = (length $ w ! 1, length $ head $ w ! 1)
            initialDeltaTotal   = initDelta hDim dDim
            (cell, deltaFunc)   = ((cata alg2_cell) . (ana coalg_cell)) (cells, fp, backProp)
            deltaTotal          = deltaFunc initialDeltaTotal

            backProp'           = BackProp (replicate dDim 0) (replicate dDim 0) (replicate (4*dDim) 0) 
                                            (replicate dDim 0) (Just $ deltaXs deltaTotal) (Just $ w )

        in  updateParameters (Layer params cell (innerLayer, tail fps, backProp')) deltaTotal

alg_cell ::  Cell (Fix Cell, [ForwardProp] -> [ForwardProp]) -> (Fix Cell, [ForwardProp] -> [ForwardProp]) -- use forwardprop storing inputs, instead of Inputs?
alg_cell InputCell = 
    (Fx InputCell, id)
alg_cell cell
    = let (nextCell, forwardProps) = (_innerCell cell)
          forwardProps' = (\fps -> 
                let fp = head fps
                    (x, label) = head (fp^.inputStack)
                    gates   = compGates (fp^.params) x (fp^.output)
                    state'  = eleadd (elemul (gates ! 3) (gates ! 2)) (elemul (gates ! 1) (fp^.prevState))
                    output' = elemul (gates ! 4) (map tanh state')
                in  ((ForwardProp gates x  (fp^.output) label output' state'  (fp^.params) (tail (fp^.inputStack))):fps)) . forwardProps
      in  (Fx (cell & innerCell .~ nextCell), forwardProps')


coalg_cell :: (Fix Cell, [ForwardProp], BackProp) -> Cell (Fix Cell, [ForwardProp], BackProp) 
coalg_cell (Fx InputCell, forwardProps, backProp)
    = InputCell
coalg_cell (Fx cell, forwardProps, backProp)
  = let fp = head forwardProps
        lastState               = (head (tail forwardProps)) ^. prevState 
        (gate, updatedState)    = (fp ^. gates, fp ^. prevState)
        (weightsW, weightsU)    = mapT2 (V.foldr (++) [[]]) (fp^.params._1, fp^.params._2)
    
        BackProp dState_next deltaError_next deltaGates_next
                f_next nextLayerDeltas nextLayerWeightsW = backProp

        deltaError =  
            case (cell, backProp ^. nextLayerWs, backProp ^. nextLayerDXs) 
            of  (EndCell {},Nothing, _)  -> (elesub (fp^.output)  (fp^.label)) 
                (Cell {},   Nothing, _)  -> eleadd deltaError_next (elesub (fp^.output)  (fp^.label))
                (_, Just w, Just dX)     -> elemul (mvmul  (transpose $ V.foldr (++) [[]] w) (head dX)) (fp ^. output) 
        
        dState     =  (elemul3 deltaError (gate ! 4) (map (sub1 . sqr . tanh) updatedState)) --{eleadd (elemul dState_next f_next)}
        deltaGates = compDGates gate deltaError dState updatedState lastState 
        (deltaX, deltaOut)     = mapT2 (mvmulk deltaGates . transpose) (weightsW, weightsU)

        deltaW     = outerProduct deltaGates (fp^.x) 
        deltaU     = 
            case cell
            of  EndCell {} -> fillMatrix (length deltaGates) (quot (length deltaGates) 4) 0.0
                Cell {}    -> outerProduct deltaGates_next (fp^.output)
        deltaB     = deltaGates

        backProp'  = backProp & nextDState .~ dState 
                            & nextDOut   .~ deltaOut 
                            & nextDGates .~ deltaGates 
                            & nextF      .~ (gate ! 1)
                            & nextLayerDXs .~ case backProp ^. nextLayerDXs 
                                              of Just dxs -> Just (tail dxs) 
                                                 Nothing  -> Nothing
    in (cell & cellState .~ updatedState
             & cellDeltas .~ (Deltas deltaW deltaU deltaB [deltaX])
             & innerCell .~ (fromJust (cell ^? innerCell), tail forwardProps, backProp'))
            
alg2_cell ::  Cell (Fix Cell, Deltas -> Deltas) ->  (Fix Cell, Deltas -> Deltas)
alg2_cell InputCell 
    = (Fx InputCell, id)
alg2_cell cell
    =   let (state, deltas, (nextCell, deltaTotalFunc)) = (_cellState cell, _cellDeltas cell, _innerCell cell)
            Deltas deltaW1 deltaU1 deltaB1 deltaXs1 = deltas
            deltaTotalFunc' = (\deltaTotal -> 
                let Deltas deltaW2 deltaU2 deltaB2 deltaXs2 = deltaTotal

                    deltaW_total = (eleaddM deltaW1 deltaW2)
                    deltaU_total = (eleaddM deltaU1 deltaU2) -- verified
                    deltaB_total = (eleadd deltaB1 deltaB2)
                    deltaXs      = deltaXs1 ++ deltaXs2
                in  Deltas deltaW_total deltaU_total deltaB_total deltaXs) . deltaTotalFunc
        in  (Fx (cell {_innerCell = nextCell}), deltaTotalFunc') --


compGates :: HyperParameters -> [Double] -> [Double] -> Gates
compGates (weightsW, weightsU, biases) x h 
    =  V.map ((replaceElement (replicate 4 sigmoid)  2 tanh ) <*>) (eleadd3v (V.map (mvmulk x) weightsW) (V.map (mvmulk h) weightsU) biases)

compDGates :: Gates -> [Double] -> [Double] -> [Double] -> [Double] -> [Double]
compDGates gate dOut dState state lastState 
    = let   d_f        = elemul4 dState lastState (gate ! 1) (map sub1 (gate ! 1))
            d_i        = elemul4 dState (gate ! 3) (gate ! 2) (map sub1 (gate ! 2))
            d_a        = elemul3 dState (gate ! 2) (map (sub1 . sqr) (gate ! 3))
            d_o        = elemul4 dOut (map tanh state) (gate ! 4) (map sub1 (gate ! 4))
      in    d_f ++ d_i ++ d_a ++ d_o

updateParameters ::  Layer k -> Deltas -> Layer k
updateParameters layer delta_total
    =   let Deltas deltaW_total deltaU_total deltaB_total deltaXs = delta_total
            (weights_w,weights_u,biases) = fromJust $ layer ^? hparams
            w = concat $ V.toList weights_w
            u = concat $ V.toList weights_u
            b = concat $ V.toList biases
            w'     = V.fromList $ map cons $ elesubm w (map2 (0.1 *) deltaW_total)
            u'     = V.fromList $ map cons $ elesubm u (map2 (0.1 *) deltaU_total)
            b'     = V.fromList $ map cons $ elesub  b (map (0.1 *)  deltaB_total)

        in layer & hparams .~ (w', u', b')



algcomp' :: (Functor f, Functor g) => (a -> Fix g) -> (f (Fix g) -> (Fix g)) -> (g a -> a) -> (f a -> a)
algcomp' h phi phi' = (cata phi') . (phi) . (fmap h)

emptyParams = ((V.fromList (replicate 4 [[]])),(V.fromList (replicate 4 [[]])),(V.fromList (replicate 4 [])))
emptyGates  = (V.fromList (replicate 4 []))

initForwardProp :: Int -> Int -> HyperParameters -> Inputs -> ForwardProp
initForwardProp h d params sample = ForwardProp (V.fromList (replicate 4 [])) [] [] [] (replicate h 0.0) (replicate h 0.0) params sample

initBackProp :: Int -> Int -> BackProp
initBackProp h d = BackProp (replicate d 0) (replicate d 0) (replicate (4*d) 0) (replicate d 0) Nothing Nothing

initDelta :: Int -> Int -> Deltas
initDelta h d = Deltas  (fillMatrix (4 * h) (d) 0.0) 
                    (fillMatrix (4 * h) (h) 0.0)
                    (replicate  (4 * h) 0.0)
                    [[]]

example =   Fx (Layer (V.fromList [[[0.7]],  [[0.95]],  [[0.45]],   [[0.6]]],
                       V.fromList [[[0.1]]      ,  [[0.8]]      ,   [[0.15]]   ,    [[0.25]]],
                       V.fromList [[0.15]       , [0.65]        , [0.2]        ,    [0.1]])
                     (Fx (EndCell [0.68381] NoDeltas (Fx (Cell [0] NoDeltas (Fx InputCell)))))
            (Fx (Layer (V.fromList [[[0.7, 0.45]],  [[0.95, 0.8]],  [[0.45, 0.25]],   [[0.6, 0.4]]],
                     V.fromList [[[0.1]]      ,  [[0.8]]      ,   [[0.15]]     ,    [[0.25]]],
                     V.fromList [[0.15]       , [0.65]         , [0.2]        ,    [0.1]])
                    (Fx (EndCell [0.68381] NoDeltas (Fx (Cell [0] NoDeltas (Fx InputCell))))) (Fx InputLayer))))

runRecurrent =  print "hi" -- $ show $ runs example



runs :: Layer k -> Layer k 
runs InputLayer = InputLayer
runs (Layer params cells innerLayer)
    = let sample = [([1,2],[0.5]),([0.5,3], [1.25])]
          dDim = length . fst $ head sample
          hDim = let (w,u,b) = params in length $ w ! 1
   
          initialForwardProp = initForwardProp hDim dDim params sample
          initialBackProp    = initBackProp hDim dDim
          initialDeltaTotal  = initDelta  hDim dDim

          (cellf, deltaTotalFunc) = let h =  (\(c, f) -> (c, f [initialForwardProp], initialBackProp))

                                    in  ((cata alg2_cell) . (meta alg_cell h coalg_cell)) cells

          deltaTotal                = deltaTotalFunc initialDeltaTotal

      in  updateParameters (Layer params cellf innerLayer) deltaTotal
