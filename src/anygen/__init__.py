import c3linearize as _c3linearize
import collections as _collections
import jinja2 as _jinja2
import json as _json
import pathlib as _pathlib
import pkg_resources as _pkg_resources
import ruamel.yaml as _yaml
from ruamel.yaml import YAML as _YAML
import runpy as _runpy
import subprocess as _subprocess
import sys as _sys
import yaql as _yaql
from .obj import Obj as _Obj
from .taskflow import TaskFlow as _TaskFlow

try:
    __version__ = _pkg_resources.get_distribution(__name__).version
except:  # noqa # pragma: no cover
    pass


ELIDE = object()


class TaggedStr(str):
    def __new__(cls, s, **kw):
        self = super().__new__(cls, s)
        self.__dict__ = kw
        return self


class TemplateResultStr(TaggedStr):
    pass


def _which(path, name):
    for path_entry in path:
        path_name = _pathlib.Path(path_entry) / name
        if path_name.exists():
            return path_name

    return None


def _checked_which(path, name):
    result = _which(path, name)
    if result is None:
        raise FileNotFoundError(name, path)
    return result


class Anygen:
    def __init__(self, produce_func):
        self._produce_func = produce_func

    def produce(self, *args, **kw):
        return self._produce_func(args + (kw,))


def _init_task_init(ctx):
    ctx.yaql = ctx.inp.yaql
    ctx.taskflow = _TaskFlow()
    ctx.extras = _collections.defaultdict(dict)

    for k, v in ctx.inp.extras.items():
        ctx.extras[k].update(v)

    def task(produce_ctx):
        produce_ctx.yaql_ctx = produce_ctx.inp.yaql_ctx

        produce_ctx.yaql_ctx["inp"] = ctx.yaql.expr.merge_dict_list.evaluate(
            context=ctx.yaql.ctx.create_child_context(),
            data=produce_ctx.inp.inp
        )

        produce_ctx.yaql_ctx["out"] = {}

    ctx.taskflow.add_task("init", func=task)


def _init_task_load_yamls(ctx):
    suffix = "anygen.yml"

    COMBINED = object()
    ROOT = object()

    loaded = {}
    yaml_loader = _YAML(typ='rt', pure=True)

    def load_list(classes):
        for i in classes:
            load(i)

    def load(class_):
        nonlocal loaded
        if class_ in loaded:
            return

        path_name = _checked_which(ctx.inp.path, class_ + "." + suffix)
        yaml = yaml_loader.load(path_name)
        extends = yaml.get("extends", [])
        # TODO proper error
        assert isinstance(extends, list)
        loaded[class_] = _Obj(path_name=path_name, yaml=yaml, extends=extends)
        load_list(extends)

    def load_root():
        path_name = _which(ctx.inp.path, suffix)
        if path_name is None:
            return
        yaml = yaml_loader.load(path_name)
        if "extends" in yaml:
            raise RuntimeError("root class cannot have 'extends'")

        loaded[ROOT] = _Obj(path_name=path_name, yaml=yaml)

    load_list(ctx.inp.classes)
    load_root()

    if ROOT in loaded:
        bases = [ROOT]
    else:
        bases = []

    def bases_func(class_):
        if class_ is COMBINED:
            return ctx.inp.classes + bases
        if class_ is ROOT:
            return []
        return list(loaded[class_].extends) + bases

    dep_graph = _c3linearize.build_graph(COMBINED, bases_func)
    c3linearization = _c3linearize.linearize(dep_graph, heads=[COMBINED])[COMBINED]
    assert c3linearization.pop(0) is COMBINED

    ctx.classes = list(reversed([loaded[i] for i in c3linearization]))

    for class_ in ctx.classes:
        class_.yaql_prefix = class_.yaml.get("marks", {}).get("yaql", "=")
        class_.elideable_key_suffix = class_.yaml.get("marks", {}).get("elide", "?")
        class_.merge_key_suffix = class_.yaml.get("marks", {}).get("merge", "+")
        class_.filter_key_sep = class_.yaml.get("marks", {}).get("filter", "|")


def _init_task_merge_yaml_consts(ctx):
    ctx.yaql.ctx0 = ctx.yaql.expr.merge_yaml_consts.evaluate(context=ctx.yaql.ctx, data=ctx.classes)
    ctx.yaql.ctx = ctx.yaql.ctx0.create_child_context()


def _init_task_py_deps(ctx):
    missing_requires = set()

    for class_ in ctx.classes:
        for req_str in class_.yaml.get('py_requires', []):
            req = _pkg_resources.Requirement.parse(req_str)
            try:
                _pkg_resources.working_set.resolve([req])
            except _pkg_resources.DistributionNotFound:
                missing_requires.add(req)

    if missing_requires:
        if _sys.base_prefix == _sys.prefix:
            raise RuntimeError(
                "missing Python requirements and not running in venv -- will not try to autoinstall",
                list(missing_requires)
            )

        _subprocess.check_call(
            [_pathlib.Path(_sys.executable).parent / "pip", "install"] +
            [str(i) for i in missing_requires]
        )


def _init_task_load_py(ctx):
    extras = ctx.extras

    for class_ in ctx.classes:
        py_fname = class_.path_name.with_suffix(".py")

        if not py_fname.exists():
            continue

        py_dict = _runpy.run_path(py_fname)

        for name, val in py_dict.items():
            if name.startswith('_'):
                continue

            split_name = name.split('_', 1)
            if len(split_name) == 1:
                continue

            name_head, name_tail = split_name
            extras[name_head][name_tail] = val


def _init_task_setup_yaql_extras(ctx):
    for k, v in ctx.extras["yaql"].items():
        ctx.yaql.ctx.register_function(_yaql.language.specs.name(k)(v))


def _init_task_setup_jinja(ctx):
    jinja_env = ctx.jinja_env = _jinja2.Environment(
        loader=_jinja2.ChoiceLoader(
            [_jinja2.DictLoader(class_.yaml.get("templates", {})) for class_ in reversed(ctx.classes)] +
            [_jinja2.FileSystemLoader(ctx.inp.path)]
        ),
        keep_trailing_newline=True
    )

    for k, v in ctx.extras["jinjafilter"].items():
        jinja_env.filters[k] = v

    for k, v in ctx.extras["jinjatest"].items():
        jinja_env.tests[k] = v

    @_yaql.language.specs.name("template")
    def yaql_template(name, **kw):
        return TemplateResultStr(jinja_env.get_template(name).render(**kw), name=name)

    ctx.yaql.ctx.register_function(yaql_template)


def yaql_ctx_to_str(ctx):
    if ctx is None:
        return '*'
    return str(dict((k, ctx.get_data(k)) for k in ctx.keys())) + ' | ' + yaql_ctx_to_str(ctx.parent)


def _init_task_prepare(ctx):
    preparers = []
    for class_ in ctx.classes:
        prepare_exprs = class_.yaml.get("prepare", [])
        if isinstance(prepare_exprs, str):
            prepare_exprs = [prepare_exprs]

        for expr in prepare_exprs:
            preparers.append(ctx.yaql.engine(expr))

    def task(produce_ctx):
        # print(yaql_ctx_to_str(produce_ctx.yaql_ctx))
        for i in preparers:
            produce_ctx.yaql_ctx = i.evaluate(context=produce_ctx.yaql_ctx, data=None)
            # print(yaql_ctx_to_str(produce_ctx.yaql_ctx))

    ctx.taskflow.add_task("prepare", func=task, after=["init"])


def _recursive_yaql(ctx, class_, data):
    def process_dict_item(k, v):
        pipeline_tail = []
        for suffix, yaql_filter in [
            (class_.elideable_key_suffix, 'elide'),
            (class_.merge_key_suffix, 'merge')
        ]:
            if k.endswith(suffix):
                k = k[:-len(suffix)]
                pipeline_tail.append(yaql_filter)

        k, *pipeline = k.split(class_.filter_key_sep)
        pipeline.extend(reversed(pipeline_tail))

        pipeline = [ctx.yaql.engine(i + '($)') for i in pipeline]

        def produce(yaql_ctx):
            result = v(yaql_ctx)

            for i in pipeline:
                result = i.evaluate(context=ctx.yaql.ctx, data=result)

            return result

        return k, produce

    def recursive_elide(data):
        if isinstance(data, dict):
            return dict((k, recursive_elide(v)) for k, v in data.items() if v is not ELIDE)
        elif isinstance(data, list):
            return list(recursive_elide(i) for i in data if i is not ELIDE)
        else:
            return data

    def recurse(data):
        if isinstance(data, str):
            if data.startswith(class_.yaql_prefix):
                expr = ctx.yaql.engine(data[len(class_.yaql_prefix):])
                return lambda yaql_ctx: recursive_elide(expr.evaluate(context=yaql_ctx))
            else:
                return lambda yaql_ctx: data
        elif isinstance(data, dict):
            items = list(process_dict_item(k, recurse(v)) for k, v in data.items())
            return lambda yaql_ctx: dict(
                (k, v)
                for k, cb in items
                for v in (cb(yaql_ctx),)
                if v is not ELIDE
            )
        elif isinstance(data, list):
            items = list(recurse(i) for i in data)
            return lambda yaql_ctx: list(
                v
                for cb in items
                for v in (cb(yaql_ctx),)
                if v is not ELIDE
            )
        else:
            return lambda yaql_ctx: data

    return recurse(data)


def _yaql_let(yaql_ctx, *args, **kw):
    yaql_ctx = yaql_ctx.create_child_context()

    for i, v in enumerate(args, 1):
        yaql_ctx[str(i)] = v

    for k, v in kw.items():
        yaql_ctx[k] = v

    return yaql_ctx


def _init_task_fragments(ctx):
    fragments = set()
    for class_ in reversed(ctx.classes):
        for k, v in class_.yaml.get("fragments", {}).items():
            if k in fragments:
                continue

            fragments.add(k)

            def one(k, v):
                frag_func = _recursive_yaql(ctx, class_, v)

                @_yaql.language.specs.name(k)
                def wrapper(*args, **kw):
                    return frag_func(_yaql_let(ctx.yaql.ctx0, *args, **kw))

                ctx.yaql.ctx0.register_function(wrapper)

            one(k, v)


def _init_task_produce(ctx):
    producers = [lambda yaql_ctx: yaql_ctx.get_data("out")] + [
        _recursive_yaql(ctx, class_, {"produce" + suffix: data})
        for class_ in ctx.classes
        for suffix in ['', class_.merge_key_suffix]
        for data in (class_.yaml.get("produce" + suffix, None),)
        if data is not None
    ]

    def task(produce_ctx):
        produce_ctx.out = ctx.yaql.expr.merge_dict_list.evaluate(
            context=ctx.yaql.ctx.create_child_context(),
            data=[i(produce_ctx.yaql_ctx) for i in producers]
        )["produce"]

    ctx.taskflow.add_task("produce", func=task, after=["prepare"])


def _init_task_result(ctx):
    def produce(inp):
        return ctx.taskflow.run(_Obj(
            yaql_ctx=ctx.yaql.ctx.create_child_context(),
            inp=inp
        ))
    ctx.out = produce


class _FunctionRegistry:
    def __init__(self):
        self.functions = []

    def __call__(self, f):
        self.functions.append(f)
        return f

    def fill_context(self, yaql_ctx):
        for f in self.functions:
            yaql_ctx.register_function(f)


_std_function_registry = _FunctionRegistry()


@_std_function_registry
def elide(x):
    if x is None or x == [] or x == {}:
        return ELIDE
    else:
        return x


def _register_std_functions():
    @_std_function_registry
    def to_yaml(data):
        return _yaml.dump(data, Dumper=_yaml.RoundTripDumper)

    @_std_function_registry
    def to_json(data):
        return _json.dumps(data)


_register_std_functions()

_YAQL_PREPARE_CTX = [
    "def(merge0, $.aggregate($1.merge_with({''=>$2}), {}).get(''))",
    "def(merge, merge0(merge0($)))"
]


class AnygenEngine:
    def __init__(self):
        self._taskflow = _TaskFlow()
        self._yaql_engine = \
            _yaql.YaqlFactory(allow_delegates=True) \
            .create(
                options={
                    # 'yaql.convertInputData': False
                    'yaql.convertInputData': True
                })
        self._yaql_ctx = _yaql.create_context(
            convention=_yaql.language.conventions.PythonConvention()
        )
        self._yaql_expr = _Obj(dict((k, self._yaql_engine(v)) for k, v in [
            ("merge_yaml_consts", "call(let, [none], $.aggregate($1.merge_with($2.yaml.get(const, {})), {}))"),
            ("merge_dict_list", "$.aggregate($1.merge_with($2), {})"),
        ]))

        _std_function_registry.fill_context(self._yaql_ctx)

        for i in _YAQL_PREPARE_CTX:
            self._yaql_ctx = self._yaql_engine(i).evaluate(context=self._yaql_ctx, data=None)

        self._taskflow.add_tasks([
            ('init', _init_task_init),
            ('load_yamls', _init_task_load_yamls),
            ('merge_yaml_consts', _init_task_merge_yaml_consts),
            ('py_deps', _init_task_py_deps),
            ('load_py', _init_task_load_py),
            ('setup_yaql_extras', _init_task_setup_yaql_extras),
            ('setup_jinja', _init_task_setup_jinja),
            ('fragments', _init_task_fragments),
            ('prepare', _init_task_prepare),
            ('produce', _init_task_produce),
            ('result', _init_task_result),
        ])

    def create(
        self,
        *,
        path,
        classes,
        extras=None
    ):
        return Anygen(self._taskflow.run(_Obj(
            yaql=_Obj(
                engine=self._yaql_engine,
                ctx=self._yaql_ctx.create_child_context(),
                expr=self._yaql_expr,
            ),
            path=path,
            classes=classes,
            extras=extras or {}
        )))


_yaql.yaqlization.yaqlize(_Obj)
_yaql.yaqlization.yaqlize(TaggedStr)
