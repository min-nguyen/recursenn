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

data HFix h xs = HRoll {hunroll :: h (HFix h) xs}