"""Vulture whitelist for intentional non-model false positives.

Pydantic model files are excluded through ``[tool.vulture].exclude``. Keep this
file limited to public runtime surfaces and protocol methods that Vulture cannot
prove are used from ``src`` alone.
"""

is_loopback  # public hosted-startup metadata
expectations  # protocol parameter on Verifier.verify

_.append_message  # assistant store public method
_.assign_unowned_sessions  # assistant store public method
_.export_tool_descriptions  # adapter inspection/debugging API
_.export_tools  # workflow executor public API
_.execute_model_output  # workflow executor public API
_.execute_model_output_async  # workflow executor public API
_.filter_tools  # registry public query API
_.finalize_expired_approvals_async  # workflow executor public API
_.find_files  # filesystem service protocol method
_.get_file_info  # filesystem service protocol method
_._iter_registered_tools  # registry internal inspection surface used by tests
_.list_directory  # filesystem service protocol method
_.list_pending_approvals  # workflow executor public API
_.preflight  # provider public API
_.pretty_json  # app presentation helper
_.read_file  # filesystem service protocol method
_.require_filesystem  # execution service protocol method
_.require_subprocess  # execution service protocol method
_.resolve_pending_approval_async  # workflow executor public API
_.resume_session_async  # harness session service public API
_.run_session_async  # harness session service public API

format_final_response  # app presentation helper
format_final_response_metadata  # app presentation helper
format_transcript_text  # app presentation helper
