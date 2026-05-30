from binaryninja import (
    BinaryView,
    Component,
    Type,
    TypeParser,
    Platform,
    Endianness,
    ArrayType,
    BackgroundTaskThread,
    demangle_gnu3,
    get_qualified_name,
    SegmentFlag,
    SectionSemantics,
)
from binaryninja.log import log_error, log_warn

import re

try:
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from ..shared_base import (
            BaseStatusHandler,
            BaseDisassemblerInterface,
            ScriptContext,
        )
        import os
        from datetime import datetime
        from typing import Literal, Union

        bv: BinaryView = None  # type: ignore
except ImportError:
    pass

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))


class BinaryNinjaDisassemblerInterface(BaseDisassemblerInterface):
    supports_fake_string_segment: bool = True

    _status: BaseStatusHandler

    _view: BinaryView
    _undo_id: str
    _components: dict[str, Component]
    _type_cache: dict[str, Type]
    _function_type_cache: dict[str, Type]

    _address_size: int
    _endianness: Literal["little", "big"]
    _address_delta: int

    TYPE_PARSER_OPTIONS = ["--target=x86_64-pc-linux", "-x", "c++", "-D_BINARYNINJA_=1"]

    def __init__(self, status: BaseStatusHandler):
        self._status = status

    def _get_or_create_type(self, type: str) -> Type:
        type = type.strip()
        if type in self._type_cache:
            return self._type_cache[type]

        parsed = self._view.get_type_by_name(type)
        if parsed is None and type.startswith(("struct ", "class ")):
            parsed = self._view.get_type_by_name(type.split(" ", 1)[1])

        if parsed is None and type.startswith("struct ") and type.endswith("__Class *"):
            parsed = self._get_or_create_type("struct Il2CppClass *")

        if parsed is None:
            parsed, name = self._view.parse_type_string(f"{type} __il2cpp_tmp")

        self._type_cache[type] = parsed
        return parsed

    def _parse_type_source(self, types: str, filename: Union[str, None] = None):
        parse_filename = filename if filename else "types.hpp"

        if hasattr(self._view, "type_container"):
            parsed_types, errors = self._view.type_container.parse_types_from_source(
                types,
                parse_filename,
                self.TYPE_PARSER_OPTIONS,
                [self.get_script_directory()],
            )
        else:
            parsed_types, errors = TypeParser.default.parse_types_from_source(
                types,
                parse_filename,
                self._view.platform
                if self._view.platform is not None
                else Platform["windows-x86_64"],
                self._view,
                self.TYPE_PARSER_OPTIONS,
            )

        if parsed_types is None:
            log_error("Failed to import types.")
            log_error(errors)
            return None

        return parsed_types

    def get_script_directory(self) -> str:
        return CURRENT_PATH

    def _is_mapped_address(self, address: int) -> bool:
        if address == 0:
            return False

        if hasattr(self._view, "is_valid_offset"):
            return self._view.is_valid_offset(address)

        for address_range in self._view.mapped_address_ranges:
            if address_range.start <= address < address_range.end:
                return True

        return False

    def _is_executable_address(self, address: int) -> bool:
        if address == 0:
            return False

        if hasattr(self._view, "is_offset_executable"):
            return self._view.is_offset_executable(address)

        for segment in self._view.segments:
            if segment.start <= address < segment.end:
                return (segment.flags & SegmentFlag.SegmentExecutable) != 0

        return False

    def _map_address(self, address: int) -> int:
        if address == 0:
            return address

        return address + self._address_delta

    def _collect_function_addresses(self, metadata: dict) -> list[int]:
        addresses = []

        def add_address(value):
            if isinstance(value, str) and value.startswith("0x"):
                try:
                    address = int(value, 0)
                except ValueError:
                    return

                if address != 0:
                    addresses.append(address)

        for address in metadata.get("functionAddresses", []):
            add_address(address)

        return addresses

    def _collect_metadata_addresses(self, metadata: dict) -> list[int]:
        addresses = []

        def add_address(value):
            if isinstance(value, str) and value.startswith("0x"):
                try:
                    address = int(value, 0)
                except ValueError:
                    return

                if address != 0:
                    addresses.append(address)

        def walk(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    if k in ("virtualAddress", "methodAddress"):
                        add_address(v)
                    else:
                        walk(v)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        for address in metadata.get("functionAddresses", []):
            add_address(address)

        walk(metadata)
        return addresses

    def _score_address_delta(self, addresses: list[int], delta: int) -> int:
        return sum(1 for address in addresses if self._is_mapped_address(address + delta))

    def _score_executable_address_delta(self, addresses: list[int], delta: int) -> int:
        return sum(1 for address in addresses if self._is_executable_address(address + delta))

    def configure_address_mapping(self, metadata: dict):
        self._address_delta = 0

        addresses = self._collect_metadata_addresses(metadata)
        function_addresses = self._collect_function_addresses(metadata)
        if len(addresses) == 0 and len(function_addresses) == 0:
            return

        addresses = addresses[:2048]
        function_addresses = function_addresses[:2048]
        raw_mapped_score = self._score_address_delta(addresses, 0)
        raw_executable_score = self._score_executable_address_delta(function_addresses, 0)

        candidates = {0}
        for address_range in self._view.mapped_address_ranges:
            if address_range.start > 0:
                candidates.add(address_range.start)

        best_delta = 0
        best_executable_score = raw_executable_score
        best_mapped_score = raw_mapped_score
        for delta in candidates:
            executable_score = self._score_executable_address_delta(function_addresses, delta)
            mapped_score = self._score_address_delta(addresses, delta)
            if (executable_score, mapped_score) > (best_executable_score, best_mapped_score):
                best_delta = delta
                best_executable_score = executable_score
                best_mapped_score = mapped_score

        minimum_function_hits = max(4, len(function_addresses) // 20)
        should_apply_delta = (
            best_delta != 0
            and best_executable_score > raw_executable_score
            and best_executable_score >= minimum_function_hits
        )

        if should_apply_delta:
            self._address_delta = best_delta
            log_warn(
                f"Applying Binary Ninja address delta 0x{self._address_delta:x} "
                f"({best_executable_score}/{len(function_addresses)} sampled functions executable, "
                f"raw executable {raw_executable_score}/{len(function_addresses)}, "
                f"{best_mapped_score}/{len(addresses)} sampled metadata addresses mapped)."
            )

    def on_start(self):
        self._view = bv  # type: ignore
        self._undo_id = self._view.begin_undo_actions()
        self._view.set_analysis_hold(True)
        self._components = {}
        self._type_cache = {}
        self._function_type_cache = {}
        self._address_delta = 0

        self._address_size = self._view.address_size
        self._endianness = (
            "little" if self._view.endianness == Endianness.LittleEndian else "big"
        )

        self._status.update_step("Parsing header")

        with open(os.path.join(self.get_script_directory(), "il2cpp.h"), "r") as f:
            parsed_types = self._parse_type_source(f.read(), "il2cpp.hpp")
            if parsed_types is None:
                return

        self._status.update_step("Importing header types", len(parsed_types.types))

        def import_progress_func(progress: int, total: int):
            self._status.update_progress(1)
            return True

        self._view.define_user_types(
            [(x.name, x.type) for x in parsed_types.types], import_progress_func
        )

    def on_finish(self):
        self._view.commit_undo_actions(self._undo_id)
        self._view.set_analysis_hold(False)
        self._view.update_analysis()

    def define_function(self, address: int, end: Union[int, None] = None):
        address = self._map_address(address)
        if self._view.get_function_at(address) is not None:
            return

        self._view.create_user_function(address)

    def define_data_array(self, address: int, type: str, count: int):
        address = self._map_address(address)
        parsed_type = self._get_or_create_type(type)
        array_type = ArrayType.create(parsed_type, count)
        var = self._view.get_data_var_at(address)
        if var is None:
            self._view.define_user_data_var(address, array_type)
        else:
            var.type = array_type

    def set_data_type(self, address: int, type: str):
        address = self._map_address(address)
        var = self._view.get_data_var_at(address)
        dtype = self._get_or_create_type(type)
        if var is None:
            self._view.define_user_data_var(address, dtype)
        else:
            var.type = dtype

    def set_function_type(self, address: int, type: str):
        address = self._map_address(address)
        function = self._view.get_function_at(address)
        if function is None:
            return

        if type in self._function_type_cache:
            function.type = self._function_type_cache[type]  # type: ignore
        else:
            self.cache_function_types([type])
            if type in self._function_type_cache:
                function.type = self._function_type_cache[type]  # type: ignore
            else:
                log_warn(
                    f"Failed to set function type at 0x{address:x}, leaving existing type: {type}"
                )

    def set_data_comment(self, address: int, cmt: str):
        address = self._map_address(address)
        self._view.set_comment_at(address, cmt)

    def set_function_comment(self, address: int, cmt: str):
        address = self._map_address(address)
        function = self._view.get_function_at(address)
        if function is None:
            return

        function.comment = cmt

    def set_data_name(self, address: int, name: str):
        address = self._map_address(address)
        var = self._view.get_data_var_at(address)
        if var is None:
            return

        if name.startswith("_Z"):
            type, demangled = demangle_gnu3(self._view.arch, name, self._view)
            var.name = get_qualified_name(demangled)
        else:
            var.name = name

    def set_function_name(self, address: int, name: str):
        address = self._map_address(address)
        function = self._view.get_function_at(address)
        if function is None:
            return

        if name.startswith("_Z"):
            type, demangled = demangle_gnu3(self._view.arch, name, self._view)
            function.name = get_qualified_name(demangled)
            # function.type = type - this does not work due to the generated types not being namespaced. :(
        else:
            function.name = name

    def add_cross_reference(self, from_address: int, to_address: int):
        from_address = self._map_address(from_address)
        to_address = self._map_address(to_address)
        self._view.add_user_data_ref(from_address, to_address)

    def import_c_typedef(self, type_def: str):
        try:
            self._view.define_user_type(None, type_def)
        except SyntaxError as e:
            log_warn(f"Failed to import type definition, skipping: {e}")

    # optional
    def _get_or_create_component(self, name: str):
        if name in self._components:
            return self._components[name]

        current = name
        if current.count("/") != 0:
            split_idx = current.rindex("/")
            parent, child = current[:split_idx], current[split_idx:]
            parent = self._get_or_create_component(name)
            component = self._view.create_component(child, parent)
        else:
            component = self._view.create_component(name)

        self._components[name] = component
        return component

    def add_function_to_group(self, address: int, group: str):
        return
        address = self._map_address(address)
        function = self._view.get_function_at(address)
        if function is None:
            return

        self._get_or_create_component(group).add_function(function)

    def _get_function_name_from_signature(self, signature: str):
        match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", signature)
        if match is None:
            return None

        return match.group(1)

    def cache_function_types(self, signatures: list[str]):
        function_sigs = set(signatures)
        if len(function_sigs) == 0:
            return

        sig_by_name = {
            self._get_function_name_from_signature(function_sig): function_sig
            for function_sig in function_sigs
        }

        typestr = ";\n".join(function_sigs).replace("this", "_this") + ";"
        parsed_types = self._parse_type_source(typestr, "cached_types.hpp")
        if parsed_types is None:
            if len(function_sigs) == 1:
                function_sig = next(iter(function_sigs))
                log_warn(f"Failed to parse function signature, skipping: {function_sig}")
                return

            for function_sig in function_sigs:
                self.cache_function_types([function_sig])
            return

        for function in parsed_types.functions:
            function_name = str(function.name)
            if function_name in sig_by_name:
                self._function_type_cache[sig_by_name[function_name]] = function.type

    # only required if supports_fake_string_segment == True
    def create_fake_segment(self, name: str, size: int) -> int:
        last_end_addr = self._view.mapped_address_ranges[-1].end
        if last_end_addr % 0x1000 != 0:
            last_end_addr += 0x1000 - (last_end_addr % 0x1000)

        self._view.memory_map.add_memory_region(
            f"mem_{name}",
            last_end_addr,
            bytes(size),
            SegmentFlag.SegmentContainsData | SegmentFlag.SegmentReadable,
        )

        self._view.add_user_section(
            name, last_end_addr, size, SectionSemantics.ReadOnlyDataSectionSemantics
        )

        return last_end_addr

    def write_string(self, address: int, value: str) -> int:
        encoded = value.encode() + b"\x00"
        self._view.write(address, encoded)
        return len(encoded)

    def write_address(self, address: int, value: int):
        address = self._map_address(address)
        self._view.write(address, value.to_bytes(self._address_size, self._endianness))


class BinaryNinjaStatusHandler(BaseStatusHandler):
    def __init__(self, thread: BackgroundTaskThread):
        self.step = "Initializing"
        self.max_items = 0
        self.current_items = 0
        self.start_time = datetime.now()
        self.step_start_time = self.start_time
        self.last_updated_time = datetime.min
        self._thread = thread

    def initialize(self):
        pass

    def update(self):
        if self.was_cancelled():
            raise RuntimeError("Cancelled script.")

        current_time = datetime.now()
        if 0.5 > (current_time - self.last_updated_time).total_seconds():
            return

        self.last_updated_time = current_time

        step_time = current_time - self.step_start_time
        total_time = current_time - self.start_time
        self._thread.progress = f"Processing IL2CPP metadata: {self.step} ({self.current_items}/{self.max_items}), elapsed: {step_time} ({total_time})"

    def update_step(self, step, max_items=0):
        self.step = step
        self.max_items = max_items
        self.current_items = 0
        self.step_start_time = datetime.now()
        self.last_updated_time = datetime.min
        self.update()

    def update_progress(self, new_progress=1):
        self.current_items += new_progress
        self.update()

    def was_cancelled(self):
        return False

    def close(self):
        pass


# Entry point
class Il2CppTask(BackgroundTaskThread):
    def __init__(self):
        BackgroundTaskThread.__init__(self, "Processing IL2CPP metadata...", False)

    def run(self):
        status = BinaryNinjaStatusHandler(self)
        backend = BinaryNinjaDisassemblerInterface(status)
        context = ScriptContext(backend, status)
        context.process()


Il2CppTask().start()
