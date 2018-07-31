{-# LANGUAGE GADTs, ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE DataKinds, TypeFamilies, TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module V where

import Prelude hiding (tail, head, replicate, map, sum, foldr, zipWith, length)
import Data.Type.Monomorphic
import Data.Singletons (Sing) 
import Data.Maybe (fromMaybe)
-- import Data.Type.Natural 
data Nat = Z | S Nat

infixl 6 :+
infixl 7 :*

type family   (n :: Nat) :+ (m :: Nat) :: Nat
type instance Z     :+ m = m
type instance (S n) :+ m = S (n :+ m)

type family   (n :: Nat) :* (m :: Nat) :: Nat
type instance Z     :* m = Z
type instance (S n) :* m = (n :* m) :+ m

data Vector a n where
  Nil  :: Vector a Z
  (:-) :: a -> Vector a n -> Vector a (S n)
infixr 5 :-

deriving instance Eq a => Eq (Vector a n)

toList :: Vector a n -> [a]
toList Nil = []
toList (x :- xs) = x : toList xs

instance Show a => Show (Vector a n) where
  showsPrec d = showsPrec d . toList

data SNat n where
  SZ :: SNat Z
  SS :: SNat n -> SNat (S n)

data Ordinal (n :: Nat) where
  OZ :: Ordinal (S n)
  OS :: Ordinal n -> Ordinal (S n)

fromList :: SNat n -> [a] -> Maybe (Vector a n)
fromList SZ     _      = Just Nil
fromList (SS n) (x:xs) = (x :-) <$> fromList n xs
fromList _      _      = Nothing

fromList' :: SingRep n => [a] -> Maybe (Vector a n)
fromList' = fromList sing

unsafeFromList :: SNat n -> [a] -> Vector a n
unsafeFromList len = fromMaybe (error "Length too short") . fromList len

-- | Unsafe version of 'unsafeFromList'.
unsafeFromList' :: SingRep n => [a] -> Vector a n
unsafeFromList' = unsafeFromList sing

-- natToSNat :: (Eq a,Num a) => a -> SNat a
-- natToSNat 0 = Z
-- natToSNat n = SS (natToSNat n)

-- ordToSNat :: Ordinal n -> SNat n
-- ordToSNat OZ = SZ
-- ordToSNat (OS (S n)) = SS (ordToSNat n)


intToNat :: (Eq a, Num a) => a -> Nat
intToNat 0 = Z
intToNat n = S (intToNat (n - 1))

sum :: Num a => Vector a n -> a
sum = foldr (+) 0

foldr :: (a -> b -> b) -> b -> Vector a n -> b
foldr _ b Nil       = b
foldr f a (x :- xs) = f x (foldr f a xs)


zipWith :: (a -> b -> c) -> Vector a n -> Vector b n -> Vector c n
zipWith _ Nil Nil             = Nil
-- zipWith _ Nil (_ :- _)        = Nil
-- zipWith _ (_ :- _) Nil        = Nil
zipWith f (x :- xs) (y :- ys) = f x y :- zipWith f xs ys

-- show
sIndex :: Ordinal n -> Vector a n -> a
sIndex OZ     (x :- _)  = x
sIndex (OS n) (_ :- xs) = sIndex n xs

replicate :: SNat n -> a -> Vector a n
replicate SZ     _ = Nil
replicate (SS n) a = a :- replicate n a

replicate' :: forall a n. SingRep n => a -> Vector a n
replicate' = replicate (sing :: SNat n)

head :: Vector a (S n) -> a
head (x :- _) = x

tail :: Vector a (S n) -> Vector a n
tail (_ :- xs) = xs

map :: (a -> b) -> Vector a n -> Vector b n
map _ Nil       = Nil
map f (x :- xs) = f x :- map f xs

length :: Vector a n -> Int
length Nil = 0
length (_ :- xs) = 1 + (length xs)

class SingRep n where
  sing :: SNat n

instance SingRep Z where
  sing = SZ

instance SingRep n => SingRep (S n) where
  sing = SS (sing :: SNat n)

data SingInstance (n :: Nat) where
  SingInstance :: SingRep n => SingInstance n

singInstance :: SNat n -> SingInstance n
singInstance SZ     = SingInstance
singInstance (SS n) =
  case singInstance n of
    SingInstance -> SingInstance

sLength :: Vector a n -> SNat n
sLength Nil = SZ
sLength (_ :- xs) = SS $ sLength xs

transpose :: SingRep n => Vector (Vector a n) m -> Vector (Vector a m) n
transpose Nil = replicate' Nil
transpose (Nil :- _) = Nil
-- show
transpose ((x :- xs) :- xss) =
  case singInstance (sLength xs) of
    SingInstance -> (x :- map head xss) :- transpose (xs :- map tail xss)
-- /show