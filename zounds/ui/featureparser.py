import ast


class ExpressionVisitor(ast.NodeVisitor):
    def __init__(self, document, locals):
        super(ExpressionVisitor, self).__init__()
        self.locals = locals
        self.document = document
        self.feature_name = None
        self.doc = None

    @property
    def result(self):
        if self.feature_name:
            feature = self.document.features[self.feature_name]
        else:
            feature = None
        return self.doc, feature

    def visit_Expr(self, node):
        if self.document is None:
            raise ValueError()

        children = list(ast.iter_child_nodes(node))
        if len(children) != 1:
            raise ValueError()

        feature_name = None

        child = children[0]
        if isinstance(child, ast.Attribute) \
                and child.attr in self.document.features:
            feature_name = child.attr
        else:
            raise ValueError()

        grandchildren = list(ast.iter_child_nodes(child))
        if len(grandchildren) != 2:
            raise ValueError()

        grandchild = grandchildren[0]
        if isinstance(grandchild, ast.Name) \
                and grandchild.id in self.locals \
                and isinstance(self.locals[grandchild.id], self.document):
            self.doc = self.locals[grandchild.id]
            self.feature_name = feature_name
        else:
            raise ValueError()


class FeatureParser(object):
    def __init__(self, document, locals):
        super(FeatureParser, self).__init__()
        self.visitor = ExpressionVisitor(document, locals)

    def parse_feature(self, statement):
        root = ast.parse(statement)
        try:
            self.visitor.visit(root)
        except ValueError:
            pass
        return self.visitor.result
