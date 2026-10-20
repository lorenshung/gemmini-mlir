# Independent current-program staged pair check

`gemmini_v26_staged_actual_pair.check_actual_staged_pair` takes two original live
`DirectSourceExecutionOwner` objects, current `source`, `proof`, `capsule_dir`,
and the original live `StagedCompletionOwner`. It launches no target process.
No success JSON, neutral callback, subclass or foreign owner can replace these
objects. It requires the same original selected target/arithmetic instances,
execution contract and observation protocol on both arms. Original methods,
code, source pins, identities and current artifacts are rechecked before and
after the operation, including an exception exit.

Both original owners freshly run `verify_source_execution`: complete current
source/ABI/input/golden binding, actual native and trace outputs, private zero
storage, every required product/readout, caller and host answer-write veto.
The source and proof must remain unchanged. Native results, CB/LLVM, selected
ELFs and complete original invocations are reopened; the two CBs must be exact.
The checker derives a staged contract from the actual arithmetic/completion
and assigned windows, rather than accepting a provider's success declaration.

The original framing parser and selected operand decoder recover actual
commands. Every selected MVIN port, including MVIN2/MVIN3, and MVOUT pointer is
normalized through unique original linked symbols. Source ABI names remain
exact; independently proved private zero objects receive a zero identity.
Actual command transport retains opcode/function/destination/xd bits, ignoring
only compiler-allocated source-register numbers. Actual full FENCE words stay
in the comparison, including predecessor/successor bits. No instruction count,
predicted schedule or normalized JSON can substitute for this original trace.

Complete tiles include the original leading fence, accumulator initialization,
all reduction commands/fences, readout configuration, complete readout and its
following fence. The comparison preserves exact prologue/suffix and entire
original tile groups. Only complete tile permutation and RHS reload omissions
are permitted. An independent per-cell ledger uses each load port's separately
selected configuration state and byte stride; other operand writes invalidate
overlapped cells. Missing first loads, changed settings, detached commands,
extra/missing fences or changed fence bits refuse. A fence does not consume a
pending PRELOAD or establish asynchronous completion.

The same actual streams are independently projected into the existing generic
`verify_staged_dense` induction. Its exact original source products, initialized
accumulator continuation and complete output ownership must hold on both arms.
This supplements the original packet proof; it does not replace it.

`gemmini_v26_staged_pair_effects.compare_decoded_pair` is a pure data helper used
inside that live path. Its structural tests supply no hardware/source authority.
In particular, a mapping dictionary cannot establish which hardware state a
port uses: only the original `SelectedLoadStreams` owner supplies that binding
to the live checker. The public function accepts no supplied decoded stream or
saved proof result.

The result is a current functional diagnostic only. It does not issue a pass
capability, prove that a particular shared pass was invoked, admit a compiler
package, qualify hardware execution or supply timing authority. The adapter
remains disabled until independently admitted actual paired execution and
compiler/library custody are available. Separate native model domain and
interval gates are unchanged. An unchanged queue pair can test the wiring;
an extra-fence queue pair should refuse the strict relation even when all its
source products and numerical outputs remain correct.
