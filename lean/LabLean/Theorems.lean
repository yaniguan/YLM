import LabLean.Semantics
/-!
# Theorems

* `check_sound`    : the checker says `true` → the protocol is valid (spec).
* `check_complete` : the protocol is valid → the checker says `true`.
* `transfer_conserves`, `step_conserves`, `run_conserves` : total liquid volume
  over all containers never changes (liquid only moves; waste is a container too).
* Well index round trip: `indexOfWell_wellOfIndex` in `LabLean.Basic`.
-/
namespace LabLean

theorem firstFail_none_iff (l : List Check) : firstFail l = none ↔ ∀ c ∈ l, c.prop := by
  induction l with
  | nil => simp [firstFail]
  | cons c l ih =>
    by_cases h : c.prop <;> simp [firstFail, h, ih]

theorem step_ok_iff (s s' : LabState) (st : Step) :
    step s st = .ok s' ↔ StepOK s st ∧ apply s st = s' := by
  unfold step StepOK
  rw [← firstFail_none_iff]
  cases firstFail (checks s st) <;> simp

/-- Checker soundness. -/
theorem check_sound (s : LabState) (l : List Step) : check s l = true → ValidProtocol s l := by
  induction l generalizing s with
  | nil => intro _; trivial
  | cons st rest ih =>
    intro h
    unfold check run at h
    cases hs : step s st with
    | error e => simp [hs] at h
    | ok s' =>
      rw [hs] at h
      have ⟨hok, happ⟩ := (step_ok_iff s s' st).1 hs
      refine ⟨hok, ?_⟩
      rw [happ]; exact ih s' h

/-- Checker completeness. -/
theorem check_complete (s : LabState) (l : List Step) : ValidProtocol s l → check s l = true := by
  induction l generalizing s with
  | nil => intro _; rfl
  | cons st rest ih =>
    intro ⟨hok, hv⟩
    have hs : step s st = .ok (apply s st) := (step_ok_iff _ _ _).2 ⟨hok, rfl⟩
    have := ih _ hv
    unfold check at this ⊢
    simp only [run, hs]
    exact this

theorem check_iff (s : LabState) (l : List Step) : check s l = true ↔ ValidProtocol s l :=
  ⟨check_sound s l, check_complete s l⟩

/-! ## Volume conservation -/

theorem total_modify (id : String) (f : Container → Container) (c : Container) :
    ∀ l, findC id l = some c → total (modifyC id f l) + c.vol.t = total l + (f c).vol.t
  | [] => by simp [findC]
  | x :: l => by
    intro h
    by_cases hx : x.id = id
    · simp [findC, hx] at h; subst h
      simp [modifyC, hx, total]; omega
    · simp [findC, hx] at h
      have := total_modify id f c l h
      simp [modifyC, hx, total]; omega

theorem findC_modify_ne (a b : String) (f : Container → Container)
    (hf : ∀ c, (f c).id = c.id) (hne : a ≠ b) :
    ∀ l, findC b (modifyC a f l) = findC b l
  | [] => rfl
  | x :: l => by
    by_cases hx : x.id = a
    · have : x.id ≠ b := by rw [hx]; exact hne
      simp [modifyC, findC, hx, hf]
      rw [← hx]; simp [this]
    · simp [modifyC, findC, hx, findC_modify_ne a b f hf hne l]

/-- A successful transfer keeps the sum of all volumes. -/
theorem transfer_conserves (s : LabState) (p : Pipette) (src dst : String) (v : Vol)
    (h : StepOK s (.transfer p src dst v)) :
    total (apply s (.transfer p src dst v)).cs = total s.cs := by
  unfold StepOK checks at h
  simp only [apply]
  cases ha : findC src s.cs with
  | none => rfl
  | some a =>
    cases hb : findC dst s.cs with
    | none => rfl
    | some b =>
      simp only [ha, hb] at h ⊢
      have hen : Enough a v := h (need (Enough a v) (.notEnough src a.vol v))
        (by simp [sourceChecks])
      have hne : src ≠ dst := h (need (src ≠ dst) (.sameContainer src)) (by simp)
      have h1 := total_modify src (takeOut v.t) a s.cs ha
      have hb' : findC dst (modifyC src (takeOut v.t) s.cs) = some b := by
        rw [findC_modify_ne src dst (takeOut v.t) (fun _ => rfl) hne]; exact hb
      have h2 := total_modify dst (putIn a v.t) b _ hb'
      simp only [takeOut, putIn] at h1 h2
      simp only [Enough] at hen
      omega

/-- Every successful step keeps the sum of all volumes
(only `transfer` moves liquid; the other steps do not touch volumes). -/
theorem step_conserves (s : LabState) (st : Step) (h : StepOK s st) :
    total (apply s st).cs = total s.cs := by
  cases st with
  | transfer p src dst v => exact transfer_conserves s p src dst v h
  | mix p c v n => simp only [apply]; cases findC c s.cs <;> rfl
  | _ => rfl

/-- A whole successful run keeps the sum of all volumes. -/
theorem run_conserves (s s' : LabState) (l : List Step) (h : run s l = .ok s') :
    total s'.cs = total s.cs := by
  induction l generalizing s with
  | nil => simp [run] at h; rw [h]
  | cons st rest ih =>
    unfold run at h
    cases hs : step s st with
    | error e => simp [hs] at h
    | ok s1 =>
      rw [hs] at h
      have ⟨hok, happ⟩ := (step_ok_iff s s1 st).1 hs
      rw [ih s1 h, ← happ, step_conserves s st hok]

end LabLean
