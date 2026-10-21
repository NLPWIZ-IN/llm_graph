import ast
import re
import unittest
from pathlib import Path
from string import Formatter

from llm_graph import constants
from llm_graph.errors import CATALOG, LLMGraphError


PACKAGE = Path(constants.__file__).parent


class ConstantTests(unittest.TestCase):
    def test_package_runtime_strings_only_live_in_constants(self):
        def inspect(node, path):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                for field, value in ast.iter_fields(node):
                    if field == "body":
                        for index, child in enumerate(value):
                            if (
                                index == 0
                                and isinstance(child, ast.Expr)
                                and isinstance(child.value, ast.Constant)
                                and isinstance(child.value.value, str)
                            ):
                                continue
                            inspect(child, path)
                    elif isinstance(value, ast.AST):
                        inspect(value, path)
                    elif isinstance(value, list):
                        for child in value:
                            if isinstance(child, ast.AST):
                                inspect(child, path)
                return
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            ):
                return
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                self.fail(f"Runtime string outside constants: {path.name}:{node.lineno}")
            if isinstance(node, ast.JoinedStr):
                self.fail(f"Inline template outside constants: {path.name}:{node.lineno}")
            for child in ast.iter_child_nodes(node):
                inspect(child, path)

        for path in PACKAGE.glob("*.py"):
            if path.name != "constants.py":
                inspect(ast.parse(path.read_text()), path)

    def test_constants_are_final_strings_and_names_are_unique(self):
        tree = ast.parse(Path(constants.__file__).read_text())
        names = []
        values = []
        for node in tree.body:
            if not isinstance(node, ast.AnnAssign):
                continue
            self.assertIsInstance(node.target, ast.Name)
            name = node.target.id
            self.assertTrue(name.isupper(), name)
            self.assertEqual(ast.unparse(node.annotation), "Final[str]")
            value = getattr(constants, name)
            self.assertIsInstance(value, str)
            names.append(name)
            values.append(value)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(values), len(set(values)))

    def test_issue_codes_and_template_contract(self):
        codes = {
            name: value for name, value in vars(constants).items()
            if name.startswith(("ERROR_", "WARNING_", "INFO_"))
        }
        self.assertEqual(len(codes), len(set(codes.values())))
        for name, code in codes.items():
            self.assertRegex(code, r"^LLMG-[EWI]\d{4}$", name)
        errors = {value for name, value in codes.items() if name.startswith("ERROR_")}
        self.assertEqual(set(CATALOG), errors)
        for code, definition in CATALOG.items():
            fields = {
                field for _, field, _, _ in Formatter().parse(definition.template)
                if field is not None
            }
            detail = {field: "example" for field in fields}
            error = LLMGraphError(code, constants.STAGE_VALIDATION, **detail)
            self.assertEqual(error.code, code)
            self.assertEqual(error.message, definition.template.format(**detail))
            self.assertEqual(error.to_dict()[constants.FIELD_CODE], code)

    def test_render_templates_use_valid_positional_placeholders(self):
        for name, template in vars(constants).items():
            if not name.startswith("TEMPLATE_"):
                continue
            fields = [
                field for _, field, _, _ in Formatter().parse(template)
                if field is not None
            ]
            self.assertTrue(all(re.fullmatch(r"\d+", field) for field in fields), name)
            count = max(map(int, fields), default=-1) + 1
            template.format(*range(count))


if __name__ == "__main__":
    unittest.main()
