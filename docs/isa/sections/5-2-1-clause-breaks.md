# 5.2.1. Clause Breaks

_(pages 49-50)_

The following conditions can break a clause: 1. VALU exception (trap) breaks a VALU clause 2. Host commands to wave (halt, resume, single step, etc) breaks all active clauses. Context-save breaks clauses of affected waves. This allows the host to read and write SGPRs & VGPRs while debugging. If clauses were not broken by host commands, the GPRs could not be read from waves other than the one currently in a clause. If a wave halts or is kill, its clauses are ended. 3. Any action that cause a wave to jump to its trap handler breaks clause (includes context-save). A wave entering HALT (including for host-initiated single-step) may break clauses.
