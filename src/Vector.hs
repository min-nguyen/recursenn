{-# LANGUAGE
     DeriveFunctor,
     DeriveFoldable,
     DeriveTraversable,
     UndecidableInstances,
     FlexibleInstances,
     GADTs,
     KindSignatures,
     TemplateHaskell #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE DataKinds, TypeFamilies, TypeOperators #-}

module Vector where
import Prelude hiding (tail, head, replicate, map, zipWith)
import Data.Functor     
import Data.Foldable
import Data.Traversable
import Data.List hiding (zipWith)
import Data.Ord
import Text.Show.Functions

data Nat = Zero | Succ Nat deriving Show

data Vector (n :: Nat) (a :: *) where
    Vnil :: Vector 'Zero a
    Vcons :: a -> Vector n a -> Vector ('Succ n) a
    -- deriving Show

type Matrix (o :: Nat) (i :: Nat) a = Vector o (Vector i a) 

-- toVector :: [a] -> Vector n a
-- toVector [] = Vnil 
-- toVector (x:xs) = Vcons x (toVector xs)

vtoList :: Vector n a -> [a]
vtoList (Vcons a v) = (a:(vtoList v))
vtoList Vnil = []


mtoList :: Matrix n m a -> [[a]]
mtoList m = vtoList (vmap vtoList m)

vhead :: Vector n a -> a
vhead (Vcons a v) = a

vtail :: Vector ('Succ n) a -> Vector n a
vtail (Vcons a v) = v 

vmap :: (a -> b) -> Vector n a -> Vector n b
vmap f (Vcons a v) = Vcons (f a) (vmap f v)
vmap f Vnil = Vnil

vzipWith :: (a -> b -> c) -> Vector n a -> Vector n b -> Vector n c
vzipWith f (Vcons a v0) (Vcons b v1) = Vcons (f a b) (vzipWith f v0 v1)
vzipWith f (Vnil) (Vnil) = Vnil

vsum :: Num a => Vector n a -> a
vsum (Vcons a v) = a + (vsum v)
vsum Vnil = 0

-- vtranspose :: Num a => Vector n 
-- type family   (n :: Nat) :+ (m :: Nat) :: Nat
-- type instance Z     :+ m = m
-- type instance (S n) :+ m = S (n :+ m)

-- type family   (n :: Nat) :* (m :: Nat) :: Nat
-- type instance Z     :* m = Z
-- type instance (S n) :* m = (n :* m) :+ m

-- data Vector a n where
--   Vnil  :: Vector a Z
--   Vcons :: a -> Vector a n -> Vector a (S n)
-- -- infixr 5 :-

-- deriving instance Eq a => Eq (Vector a n)

-- vtoList :: Vector a n -> [a]
-- vtoList Vnil = []
-- vtoList (Vcons x xs) = x : vtoList xs

-- instance Show a => Show (Vector a n) where
--   showsPrec d = showsPrec d . vtoList

-- data SNat n where
--   SZero :: SNat Zero
--   SSucc :: SNat n -> SNat (Succ n)

vsnoc :: a -> Vector n a -> Vector ('Succ n) a
vsnoc a (Vcons a' v') = Vcons a' (vsnoc a v')
vsnoc a Vnil = Vcons a Vnil

-- vreplicate :: SNat n -> a -> Vector n a
-- vreplicate SZero     _ = Vnil
-- vreplicate (SSucc n) a = Vcons a (vreplicate n a)

-- vreplicate' :: forall a n. SingRep n => a -> Vector n a
-- vreplicate' = vreplicate (sing :: SNat n)

-- vlength :: Vector n a -> Int
-- vlength Vnil = 0
-- vlength (Vcons x xs) = 1 + (vlength xs)

-- class SingRep n where
--   sing :: SNat n

-- instance SingRep Zero where
--   sing = SZero

-- instance SingRep n => SingRep (Succ n) where
--   sing = SSucc (sing :: SNat n)

-- data SingInstance (n :: Nat) where
--   SingInstance :: SingRep n => SingInstance n

-- singInstance :: SNat n -> SingInstance n
-- singInstance SZero     = SingInstance
-- singInstance (SSucc n) =
--   case singInstance n of
--     SingInstance -> SingInstance

-- sLength :: Vector n a -> SNat n
-- sLength Vnil = SZero
-- sLength (Vcons _ xs) = SSucc $ sLength xs

-- vtranspose :: SingRep n => Vector m (Vector n a)  -> Vector n (Vector m a) 
-- vtranspose Vnil = vreplicate' Vnil
-- vtranspose (Vcons Vnil _) = Vnil
-- -- show
-- vtranspose (Vcons (Vcons x  xs) xss) =
--   case singInstance (sLength xs) of
--     SingInstance -> Vcons (Vcons x (vmap vhead xss)) (vtranspose (Vcons xs  (vmap vtail xss)))
-- -- /show
