"""Bind the logical kernel ABI and selected native completion entrypoints."""
import re

_ENTRY = re.compile(r'\bvoid\s+gemmini_kernel\s*\(([^;{}]*)\)\s*\{')
_COMPLETION = r'''__asm__(".pushsection .text.kernel_complete,\"ax\",@progbits\n.weak kernel_complete\n.type kernel_complete,@function\nkernel_complete:\nfence\nret\n.popsection\n");'''
_LOGICAL_ALIAS = r'''__asm__(".globl gemmini_kernel\n.set gemmini_kernel,kernel_entry\n");'''

def copied_entry_commands(llvm_ir):
    """Detect custom commands cloned into the generated weak C wrapper.

    This inspects our stock clang output, not an input program. A linked device
    group can discard the weak wrapper, so its body must not contain a second
    copy of the logical kernel's accelerator instructions.
    """
    body=re.search(r'^define[^\n]*@kernel_entry\([^\n]*\)[^{\n]*\{\n(.*?)^}',
                   llvm_ir,re.MULTILINE|re.DOTALL)
    return body is not None and '.insn' in body.group(1)

def prevent_entry_copy(source):
    found=list(_ENTRY.finditer(source))
    if len(found)!=1:raise ValueError('expected one logical kernel to protect')
    start=found[0].start()
    return source[:start]+'__attribute__((noinline)) '+source[start:]

def adapt_source(source, whole_program=False):
    """Supply native symbols while keeping the logical pointer convention.

    Primitive kernels use a weak native wrapper, so multiple separately named
    model kernels link together. A whole program has one LLVM function; its
    logical symbol is an assembler alias, without a forwarding stack frame.
    """
    found=list(_ENTRY.finditer(source))
    if len(found)!=1:raise ValueError('expected one generated logical kernel')
    entry=found[0]
    params=entry.group(1)
    if whole_program:
        start=entry.start()
        symbol=source.index('gemmini_kernel',start)
        source=source[:start]+'__attribute__((weak)) '+source[start:symbol]+'kernel_entry'+source[symbol+len('gemmini_kernel'):]
        source+='\n'+_LOGICAL_ALIAS
    else:
        args=[]
        for param in params.split(','):
            if not param.strip() or param.strip()=='void':continue
            name=re.search(r'([A-Za-z_][A-Za-z0-9_]*)\s*$',param)
            if name is None:raise ValueError('unsupported generated pointer declaration')
            args.append(name.group(1))
        source+='\n__attribute__((weak,noinline)) void kernel_entry('+params+') { gemmini_kernel('+','.join(args)+'); }\n'
    return source+'\n'+_COMPLETION+'\n'
