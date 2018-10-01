{-# LANGUAGE
     DeriveFunctor,
     DeriveFoldable,
     DeriveTraversable,
     TypeFamilies,
     GADTs,
     DataKinds,
     ExistentialQuantification,
     RankNTypes,
     ScopedTypeVariables #-}

module Dnn where

import Utils
import Data.Functor     
import Data.Foldable
import Data.Maybe
import Data.Traversable
import Data.List
import Data.Ord
import Text.Show.Functions
-- import qualified Data.Vector.Sized as V
import qualified Vector as V
import Data.Number.Nat
import Debug.Trace
import Control.Lens
import Data.Maybe
import Control.Lens.Tuple
---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                        Fully Connected NN                      | | ---
  --- ‾------------------------------------------------------------------‾---


-- type Weights            = [[Double]]
type Biases             = [Double]
type Inputs             = [Double]
type Outputs            = [Double]
type Weights            = [[Double]]
type Activation         =  Double  ->  Double
type Activation'        =  Double  ->  Double
type LossFunction       = [Double] -> [Double] -> Double
type DesiredOutput      = [Double]
type FinalOutput        = [Double]
type Deltas             = [Double]

-- newtype Fox n m f = Fox (V.S' n) (V.S' m) (f (Fox (V.S' m) (V.S' a) ))

data Fox n m f = forall i. V.Nat_ i => Fox n m (f (Fox m i f)) | Term (f (Fox m n f))
    

data Layer k = 
        Layer   { 
                  _weights      :: Weights,
                  _biases       :: Biases,
                  _activation   :: Activation,
                  _activation'  :: Activation',
                  _nextLayer    :: k
                } 
    |   InputLayer  deriving (Show, Functor, Foldable, Traversable)

example :: (V.Nat_ n, V.Nat_ m) => (V.S' n) -> (V.S' m) -> Fox (V.S' n) (V.S' m) Layer
example n m =  (Fox n m (Layer [[]] [] sigmoid sigmoid' (Term InputLayer ) ))
