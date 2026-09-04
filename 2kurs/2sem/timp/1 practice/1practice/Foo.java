public class Foo {
	private int x;
	private int y;
	public Foo() { x=0;y=0; }
	public Foo (int _x, int _y) {x=_x; y=_y;}

	@Override
	public String toString() {
		return "Foo(x=)" + x + ", y= " + y + ")";
	}
}
