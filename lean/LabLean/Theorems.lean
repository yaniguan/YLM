import LabLean.Spec
import LabLean.Checker
/-!
# Theorems

* `checks_iff`     : for one step, the Bool checks pass ↔ `StepOK` (the spec).
* `initChecks_iff` : the start-state checks pass ↔ `WellFormed`.
* `check_sound`, `check_complete`, `check_iff` : `check s l = true ↔ ValidProtocol s l`.
  `Decidable (ValidProtocol s l)` is built from `check_iff`, so `decide` runs the checker.
* `step_conserves`, `steps_conserve`, `run_conserves` : total liquid volume never changes.
* `step_fits`, `steps_fit`, `valid_fits` : no container ever goes over its capacity.
-/
namespace LabLean

/-! ## Bool tests ↔ spec rules -/

theorem firstErr_none_iff (l : List (Bool × Error)) :
    firstErr l = none ↔ ∀ x ∈ l, x.1 = true := by
  induction l with
  | nil => simp [firstErr]
  | cons x l ih => obtain ⟨b, e⟩ := x; cases b <;> simp [firstErr, ih]

theorem nodupB_iff (l : List String) : nodupB l = true ↔ l.Nodup := by
  induction l with
  | nil => simp [nodupB]
  | cons x l ih => simp [nodupB, ih, List.nodup_cons]

theorem findC_spec {id : String} {c : Container} :
    ∀ {l}, findC id l = some c → c ∈ l ∧ c.id = id
  | [] => by simp [findC]
  | x :: l => by
    unfold findC; split
    · intro h; cases h; simp_all
    · intro h; have := findC_spec h; simp_all

theorem findC_isSome_iff (id : String) :
    ∀ l, (findC id l).isSome ↔ ∃ c ∈ l, c.id = id
  | [] => by simp [findC]
  | x :: l => by
    unfold findC; split
    · simp_all
    · simp [findC_isSome_iff id l]; intro h; simp_all

@[simp] theorem hasTipB_iff (s : LabState) : hasTipB s = true ↔ HasTip s := by
  unfold hasTipB HasTip; cases s.tip <;> simp

@[simp] theorem sealedB_iff (s : LabState) (c : Container) : sealedB s c = true ↔ IsSealed s c := by
  unfold sealedB IsSealed; cases c.plate <;> simp

@[simp] theorem diluentB_iff (s : LabState) (r : String) : diluentB s r = true ↔ IsDiluent s r := by
  simp [diluentB, IsDiluent]

@[simp] theorem tipCleanB_iff (s : LabState) (c : Container) : tipCleanB s c = true ↔ TipClean s c := by
  simp [tipCleanB, mayEnterB, TipClean, MayEnter, NotMoreConc]

@[simp] theorem knownB_iff (s : LabState) (x : String) : knownB s x = true ↔ Known s x := by
  simp [knownB, Known, IsPlate, findC_isSome_iff]

theorem thawChecks_iff (s : LabState) (c : Container) :
    (∀ x ∈ thawChecks s c, x.1 = true) ↔ Thawed s c := by
  simp only [thawChecks, List.mem_map, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂,
    Thawed]
  simp [Decidable.imp_iff_not_or]

@[simp] theorem hasTipB_false (s : LabState) : hasTipB s = false ↔ s.tip = .none := by
  unfold hasTipB; cases s.tip <;> simp

@[simp] theorem sealedB_false (s : LabState) (c : Container) :
    sealedB s c = false ↔ ¬ IsSealed s c := by
  rw [← sealedB_iff, Bool.not_eq_true]

theorem enterChecks_iff (s : LabState) (p : Pipette) (id : String) (a : Container) (v : Vol) :
    (∀ x ∈ enterChecks s p id a v, x.1 = true) ↔
    HasTip s ∧ InRange p v ∧ ¬ IsSealed s a ∧ TipClean s a ∧ Enough a v ∧ Thawed s a := by
  simp only [enterChecks, List.mem_append, or_imp, forall_and, thawChecks_iff]
  simp [InRange, Enough, and_assoc]

/-- The Bool checks of one step hold exactly when the spec `StepOK` holds. -/
theorem checks_iff (s : LabState) (st : Step) :
    (∀ x ∈ checks s st, x.1 = true) ↔ StepOK s st := by
  cases st with
  | pickTip => simp [checks]; exact ⟨.pickTip, fun h => by cases h; assumption⟩
  | dropTip => simp [checks]; exact ⟨.dropTip, fun h => by cases h; assumption⟩
  | transfer p src dst v =>
    cases ha : findC src s.cs with
    | none => simp [checks, ha]; intro h; cases h; simp_all
    | some a =>
      cases hb : findC dst s.cs with
      | none => simp [checks, ha, hb]; intro h; cases h; simp_all
      | some b =>
        simp only [checks, ha, hb, List.mem_cons, List.mem_append, or_imp, forall_and,
          enterChecks_iff]
        simp
        constructor
        · intro h; refine .transfer ha hb ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;> simp_all [NoOverflow]
        · intro h; cases h; simp_all [NoOverflow]
  | mix p c v n =>
    cases ha : findC c s.cs with
    | none => simp [checks, ha]; intro h; cases h; simp_all
    | some a =>
      simp only [checks, ha, enterChecks_iff]
      constructor
      · intro h; refine .mix ha ?_ ?_ ?_ ?_ ?_ ?_ <;> simp_all
      · intro h; cases h; simp_all
  | incubate x t time =>
    simp [checks]
    exact ⟨fun ⟨h1, h2, h3⟩ => .incubate h1 ⟨h2, h3⟩, fun h => by cases h; simp_all [SafeTemp]⟩
  | centrifuge rotor rpm time =>
    simp only [checks, List.mem_append, List.mem_map, List.mem_range,
      or_imp, forall_and, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂]
    simp [nodupB_iff, or_imp, forall_and, ← Decidable.imp_iff_not_or]
    constructor
    · intro h; refine .centrifuge ?_ ?_ ?_ ?_ ?_ ?_ ?_ <;> simp_all [Balanced, IsPlate]
    · intro h; cases h; simp_all [Balanced, IsPlate]
  | «seal» p =>
    simp [checks]
    exact ⟨fun ⟨h1, h2⟩ => .seal h1 h2, fun h => by cases h; simp_all [IsPlate]⟩
  | thaw r =>
    simp [checks]
    exact ⟨fun h => .thaw h, fun h => by cases h; assumption⟩

theorem plateKnownB_iff (s : LabState) (c : Container) :
    plateKnownB s c = true ↔ ∀ p, c.plate = some p → p ∈ plateIds s := by
  unfold plateKnownB; cases c.plate <;> simp

theorem initChecks_iff (s : LabState) : (∀ x ∈ initChecks s, x.1 = true) ↔ WellFormed s := by
  simp only [initChecks, List.mem_cons, List.mem_append, List.mem_map, or_imp, forall_and,
    forall_exists_index, and_imp, forall_apply_eq_imp_iff₂, forall_eq, plateKnownB_iff]
  simp only [nodupB_iff, decide_eq_true_eq, Bool.not_eq_eq_eq_not, Bool.not_true]
  constructor
  · rintro ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩; exact ⟨h1, h2, by simpa using h3, h4⟩
  · rintro ⟨h1, h2, h3, h4⟩; exact ⟨⟨⟨h1, h2⟩, by simpa using h3⟩, h4⟩

/-! ## Checker ↔ spec -/

theorem step_ok_iff (s s' : LabState) (st : Step) :
    step s st = .ok s' ↔ StepOK s st ∧ apply s st = s' := by
  unfold step
  rw [← checks_iff, ← firstErr_none_iff]
  cases firstErr (checks s st) <;> simp

theorem runSteps_ok_iff (s s' : LabState) (l : List Step) :
    runSteps s l = .ok s' ↔ StepsOK s l ∧ l.foldl apply s = s' := by
  induction l generalizing s with
  | nil => simp [runSteps]; exact fun _ => .nil
  | cons st rest ih =>
    simp only [runSteps]
    cases hs : step s st with
    | error e =>
      simp only [reduceCtorEq, false_iff, not_and]
      intro h; cases h with
      | cons hok _ => exact absurd ((step_ok_iff s _ st).2 ⟨hok, rfl⟩) (by simp [hs])
    | ok s1 =>
      have ⟨hok, happ⟩ := (step_ok_iff s s1 st).1 hs
      subst happ
      rw [ih]
      constructor
      · intro ⟨h1, h2⟩; exact ⟨.cons hok h1, h2⟩
      · intro ⟨h1, h2⟩; cases h1 with | cons _ h => exact ⟨h, h2⟩

/-- Checker soundness: if the checker accepts, the protocol meets the spec. -/
theorem check_sound (s : LabState) (l : List Step) : check s l = true → ValidProtocol s l := by
  unfold check run
  cases hi : firstErr (initChecks s) with
  | some e => simp
  | none =>
    cases hr : runSteps s l with
    | error e => simp
    | ok s' =>
      intro _
      exact ⟨(initChecks_iff s).1 ((firstErr_none_iff _).1 hi),
             ((runSteps_ok_iff s s' l).1 hr).1⟩

/-- Checker completeness: every protocol that meets the spec is accepted. -/
theorem check_complete (s : LabState) (l : List Step) : ValidProtocol s l → check s l = true := by
  intro ⟨hwf, hsteps⟩
  have hi : firstErr (initChecks s) = none := (firstErr_none_iff _).2 ((initChecks_iff s).2 hwf)
  have hr := (runSteps_ok_iff s _ l).2 ⟨hsteps, rfl⟩
  simp [check, run, hi, hr]

theorem check_iff (s : LabState) (l : List Step) : check s l = true ↔ ValidProtocol s l :=
  ⟨check_sound s l, check_complete s l⟩

/-- `decide` on the spec runs the checker; `check_iff` makes this correct. -/
instance (s : LabState) (l : List Step) : Decidable (ValidProtocol s l) :=
  decidable_of_iff _ (check_iff s l)

instance (s : LabState) (st : Step) : Decidable (StepOK s st) :=
  decidable_of_iff _ ((firstErr_none_iff _).trans (checks_iff s st))

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

/-- A valid step keeps the sum of all volumes (waste is a container too; `mix` moves nothing). -/
theorem step_conserves (s : LabState) (st : Step) (h : StepOK s st) :
    total (apply s st).cs = total s.cs := by
  cases h with
  | @transfer src a dst b _ _ v ha hb hne _ _ _ _ _ _ hen _ =>
    simp only [apply, ha, hb]
    have h1 := total_modify src (takeOut v.t) a s.cs ha
    have hb' : findC dst (modifyC src (takeOut v.t) s.cs) = some b := by
      rw [findC_modify_ne src dst (takeOut v.t) (fun _ => rfl) hne]; exact hb
    have h2 := total_modify dst (putIn a v.t) b _ hb'
    simp only [takeOut, putIn] at h1 h2
    simp only [Enough] at hen
    omega
  | @mix c _ _ _ _ _ ha => simp only [apply, ha]
  | _ => rfl

/-- A valid protocol keeps the sum of all volumes. -/
theorem steps_conserve (s : LabState) (l : List Step) (h : StepsOK s l) :
    total (l.foldl apply s).cs = total s.cs := by
  induction h with
  | nil => rfl
  | cons hok _ ih => simp only [List.foldl]; rw [ih, step_conserves _ _ hok]

/-- An accepted protocol keeps the sum of all volumes. -/
theorem run_conserves (s s' : LabState) (l : List Step) (h : run s l = .ok s') :
    total s'.cs = total s.cs := by
  unfold run at h
  cases hi : firstErr (initChecks s) with
  | some e => simp [hi] at h
  | none =>
    simp only [hi] at h
    have ⟨hok, hfin⟩ := (runSteps_ok_iff s s' l).1 h
    rw [← hfin]; exact steps_conserve s l hok

/-! ## No container ever overflows -/

def Fits (cs : List Container) : Prop := ∀ c ∈ cs, c.vol.t ≤ c.cap.t

theorem mem_modifyC (id : String) (f : Container → Container) (c : Container) :
    ∀ l, c ∈ modifyC id f l → c ∈ l ∨ ∃ c0, findC id l = some c0 ∧ c = f c0
  | [] => by simp [modifyC]
  | x :: l => by
    unfold modifyC findC
    by_cases hx : x.id = id
    · simp only [hx, ite_true, List.mem_cons]
      rintro (h | h)
      · exact .inr ⟨x, rfl, h⟩
      · exact .inl (.inr h)
    · simp only [hx, ite_false, List.mem_cons]
      rintro (h | h)
      · exact .inl (.inl h)
      · rcases mem_modifyC id f c l h with h' | h'
        · exact .inl (.inr h')
        · exact .inr h'

theorem step_fits (s : LabState) (st : Step) (h : StepOK s st) (hf : Fits s.cs) :
    Fits (apply s st).cs := by
  cases h with
  | @transfer src a dst b _ _ v ha hb hne _ _ _ _ _ _ hen hfit =>
    simp only [apply, ha, hb]
    intro c hc
    have hb' : findC dst (modifyC src (takeOut v.t) s.cs) = some b := by
      rw [findC_modify_ne src dst (takeOut v.t) (fun _ => rfl) hne]; exact hb
    rcases mem_modifyC _ _ _ _ hc with hc | ⟨c0, h0, rfl⟩
    · rcases mem_modifyC _ _ _ _ hc with hc | ⟨c1, h1, rfl⟩
      · exact hf c hc
      · rw [ha] at h1; cases h1
        have := hf a (findC_spec ha).1
        simp only [takeOut]; omega
    · rw [hb'] at h0; cases h0
      simpa [putIn, NoOverflow] using hfit
  | @mix c _ _ _ _ _ ha => simp only [apply, ha]; exact hf
  | _ => exact hf

/-- In a valid protocol, every container stays within its capacity at every step. -/
theorem steps_fit (s : LabState) (l : List Step) (h : StepsOK s l) (hf : Fits s.cs) :
    Fits (l.foldl apply s).cs := by
  induction h with
  | nil => exact hf
  | cons hok _ ih => exact ih (step_fits _ _ hok hf)

theorem valid_fits (s : LabState) (l : List Step) (h : ValidProtocol s l) :
    Fits (l.foldl apply s).cs :=
  steps_fit s l h.2 h.1.withinCap

end LabLean
