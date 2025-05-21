# Strange Loop - Simple Made Easy (2011)
### https://www.youtube.com/watch?v=SxdOUGdseq4
> Rich Hickey
> Simplicity is the ultimate sophistication. - Leonardo da Vinci

## Simple
* **OBJECTIVE** - can look at and determine it is not interleaved and preforms just one task
* One braid (original meaning)
* One task
* One concept
* One role
* One dimension
* NOT just 1 instance
* Not just 1 operation
* Can have multiple instances, just not interleaved (interdependent)

## Easy
* **RELATIVE** - playing the violin and reading German are hard for some, and easy for others
* Near, at hand
    * On hard drive, IDE, toolset, apt get, npm install
* Familiar, near to OUR UNDERSTANDING
    * _If you want everything to be FAMILIAR, you won't learn anything new_
    * German is hard...well, I don't speak German, it's not familiar.
* Near our capabilities
    * I don't play violin well, but that's ok because I don't play the violin
    * However, in our industry, don't want to hurt our egos so we don't talk about it
        * Not that embarrassing to talk about though as we aren't all that different in this space

## Construct vs Artifact
* Focus on experience of the use of the CONSTRUCT...
    * **Program with Constructs**: languages, libraries
        * These all have characteristics in them built-in
    * **Business of Artifacts**: ship software (artifacts), run, change, update, performance 
        * These are attributes of the ARTIFACT, **NOT** the original construct
        * Yet we focus on the CONSTRUCT
            * Programmer Convenience: 'Look how easy it is to write the code here! No semi-colons!'
            * Programmer Replace-ability
* ...Rather than the long term results of use
    * Software quality, correctness
    * Maintenance change
* _We must assess constructs by their artifacts_

## Limits
* We can only hope to make things reliable IF we UNDERSTAND those things
* We can only UNDERSTAND a few things at a time
* Intertwined/interleaved/braided things MUST be UNDERSTOOD TOGETHER
* _Complexity Undermines UNDERSTANDING_

## Change - _New Capabilities_
* Changes to software require analysis and decisions
* What will be impacted?
* Where do changes need to be made?
* Your ability to UNDERSTAND and THINK/REASON about your software is critical to changing it without fear

## Debugging - _Fixing Issues_
* True of every bug? _It passed all the tests_
* Tests are like guardrails, they catch some things, but we don't drive our cars purposefully hitting guardrails

## Development Speed
* Easy stuff is making my life fast - _Early speed_
    * Developers have solved the speed issue that sprinters fail: Fire the starting pistol every 100 yards and call it a new sprint! :) 
    * Ignoring complexity will slow you down over the long haul
    * Will make each sprint more and more about fixing issues you built in from the beginning

## Easy Yet Complex?
* Many complicating constructs (language, tools) are:
    * Succinctly described
    * Familiar
    * Available
    * Easy to Use
* _What matters is the complexity the ARTIFACT/SOFTWARE the CONSTRUCTS yield/build_
    * Any such complexity is _incidental_

## Benefits of Simplicity
* Ease of understanding
* Ease of change
* Easier debugging
* Flexibility (modularity) - policy, locations
* _Will having test suites and refactoring tools make changing a knitted castle faster than a lego castle? No way._

## How do we make things easy?
* _Location_: Bring to hand by installing (and getting it APPROVED for use)
* _Familiar_: learning, trying it out
* _Mental Capability_: Much harder here. Can't free up a ton of mental capacity, need to make things near by simplifying them (juggling maximum of humans)

## Parens are Hard!
* _Location_: Not at hand for most people (unless you have the right editor or tool that helps with them)
* _Familiar_: Nor familiar with them
* _Familiar_: But are they simple? In some cases NO (CL/Scheme). Overloaded complexity
* _Issue is with the language (CONSTRUCT) - poor design_
* **COULD BE FIXED**: adding a data structure for grouping

## Always hear about how things are easier or shorter, but what were the downsides of making that decision?
| Complexity | Simplicity (NOT entangled) |
| --- | --- |
| State, Objects | Values |
| Methods (in Objects/Classes) | Functions (stand alone), Namespaces |
| vars | managed refs |
| inheritance, switch, matching | Polymorphism a la carte |
| Syntax | Data |
| Imperative loops, fold | Set functions |
| Actors | Queues |
| ORM | Declarative data manipulation |
| Conditionals | Rules |
| Inconsistency | Consistency |

## Complect
* To interleave, entwine, braid
* Complecting things is the source of complexity :)

## Compose
* To place together (Legos)
* Composing SIMPLE components is the key to robust software

## Modularity and Simplicity
* You can write software that is modular, but completely complected to other things
* What do we want to allow these things to think about? The interface / connections
* Don't be fooled by code organization

## State is NEVER Simple
* Complects value and time (you don't get a value without knowing time)
* This complexity is EASY, it's in everything
* Everytime you ask a question you get a different answer
* Not mitigated by modules, encapsulation
* Recreating state when something went bad...super hard to do

## Not all refs/vars are Equal
* Unfortunately, none of these make state simple
* They WARN of state and help reduce it

## The Complexity Toolkit
| Construct | Complects |
| --- | --- | 
| State | Everything that touches it |
| Objects | State, identity, value |
| Methods | Function and state, namespaces (derive from 2 things in Java that have the same method)|
| Syntax | Meaning, order | 
| Inheritance | Types | 
| Switch/Matching | Multiple who / what pairs |
| variables | value, time (cannot recover a composite mutable (changeable) thing) |
| Imperative loops, fold | what / how |
| Actors | what / who |
| ORM | OMG (multiple meanings of *value*) |
| Conditionals | why, rest of program |

## The Simplicity Toolkit
| Construct | Get it by... |
| --- | --- |
| Values | final, val, persistent, immutable, constant |
| Functions | a.k.a. stateless methods |
| Namespaces | language/CONSTRUCT Dependent |
| Data | maps, arrays, sets, XML, JSON | 
| Polymorphism a la carte | Protocols, type classes |
| Set functions | Libraries |
| Queues | Libraries |
| Declarative Data Manipulation | SQL |
| Rules | Libraries, Prolog (instead of conditionals) |
| Consistency | Transactions, values |

## Environmental Complexity
* Resources, memory, CPU
* Inherent complexity in implementation space (can't change things the customer has setup)
* Segmentation is possible, multi-threading (but can add waste/complexity/maintenance)
* Individual policies for each application (adds complexity, maintenance)
> Programming, boiled down, is effective thinking to avoid complexity and separate concerns

## Abstraction for Simplicity
* Building your own constructs
* _Abstract_ - drawn away (from physical nature of thing)
* I DON'T know, I DON'T WANT to know
* Who, what, when, where, why, and how (to break things apart, NOT *how* to do something)

### WHAT
* Operations
* Form abstractions from related sets of functions
* INTERFACE of inputs, outputs, semantics
* **HOW only COMPLECTS things** - someone else will do the how

### WHO
* Entities implementing abstractions
* Build from sub-components *direct-injection* style (not hardwired, take as arguments)
* More sub-components and fewer interfaces
* **Details around how the component and relationships with other entities COMPLECTS things**

### HOW
* Implementing logic
* Connect all abstractions using the polymorphism constructs rather than switch statements
* Use abstractions that allow the final implementor to define the **HOW** (no tying of hands)
* All these implementations should be islands or you COMPLECT things

## WHEN / WHERE
* Avoid using directly connected objects
* If thing A has to call thing B, you've COMPLECTED things
    * Use queues

## WHY
* Policy and rules of the application
* Often, strewn EVERYWHERE
* Try and define this OUTSIDE

## Information _IS_ Simple
* Don't ruin it
* By hiding it behind a micro-language
    * i.e. a class with information-specific methods
* Ties logic to representation layer
* Represent DATA AS DATA


## Simplifying
1. Identify individual threads/roles/dimensions
2. Follow through the user story/code
3. Disentangling
* Choose **SIMPLE** constructs over complexity-generating constructs
    * It's the ARTIFACTS, not the authoring
* Abstractions with **SIMPLICITY** as a basis
* Take more time to **SIMPLIFY** the problem space before you start
* **Simplicity** often means making more things, not fewer


# Appendix - Functions vs Methods
 
## Key Differences

* Association:
    * Functions are independent
    * Methods are associated with objects/classes
* Access to Data:
    * Functions can only access data passed to them as parameters
    * Methods can access the object's internal state
* Calling Syntax:
    * Functions: function_name(arguments)
    * Methods: object.method_name(arguments) or class.method_name(arguments)
* Context:
    * Functions operate in a global context
    * Methods operate in the context of their object/class
* First Parameter:
    * Functions can have any parameters
    * Methods automatically receive the object instance as their first parameter (usually called self in Python)

* Here's a more comprehensive example showing both:
```
# Function
def greet(name):
    return f"Hello, {name}!"

# Class with methods
class Person:
    def __init__(self, name):
        self.name = name
    
    def greet(self):  # Method
        return f"Hello, my name is {self.name}"
    
    @classmethod
    def create_anonymous(cls):  # Class method
        return cls("Anonymous")
    
    @staticmethod
    def say_hello():  # Static method (technically a function)
        return "Hello!"

# Using the function
print(greet("Alice"))  # Output: Hello, Alice!

# Using the class and its methods
person = Person("Bob")
print(person.greet())  # Output: Hello, my name is Bob

# Using class method
anon = Person.create_anonymous()
print(anon.greet())  # Output: Hello, my name is Anonymous

# Using static method
print(Person.say_hello())  # Output: Hello!
```

## Special Types of Methods:

* Instance Methods:
    * Regular methods that operate on instance data
    * First parameter is self
    * Can access and modify instance data
* Class Methods:
    * Marked with @classmethod
    * First parameter is cls
    * Can access and modify class data
    * Can be called on the class itself
* Static Methods:
    * Marked with @staticmethod
    * No automatic first parameter
    * Can't access instance or class data
    * Essentially functions that live in the class namespace
* Property Methods:
    * Marked with @property
    * Allow you to access methods like attributes
    * Useful for computed properties

# Appendix - Subcomponents

> A subcomponent in programming is a smaller, self-contained piece of a larger component or system that performs a specific function or represents a specific part of the user interface. Let me break this down with examples:

## Key Characteristics of Subcomponents:
* Modularity:
    * Self-contained units of code
    * Can be reused across different parts of an application
    * Have their own state and logic
* Hierarchy:
    * Exist within a parent component
    * Can have their own subcomponents
    * Form a tree-like structure
* Encapsulation:
    * Hide internal implementation details
    * Expose a clear interface
    * Manage their own state

## Examples

### Python
```
# Parent Service
class OrderService:
    def __init__(self):
        self.payment_processor = PaymentProcessor()  # Subcomponent
        self.inventory_manager = InventoryManager()  # Subcomponent
        self.notification_service = NotificationService()  # Subcomponent

    def process_order(self, order):
        self.payment_processor.process_payment(order)
        self.inventory_manager.update_stock(order)
        self.notification_service.send_confirmation(order)

# Subcomponent
class PaymentProcessor:
    def process_payment(self, order):
        # Payment processing logic
        pass
```

### React
```
// Parent Component
function UserProfile() {
  return (
    <div className="profile">
      <UserHeader />      {/* Subcomponent */}
      <UserDetails />     {/* Subcomponent */}
      <UserSettings />    {/* Subcomponent */}
    </div>
  );
}

// Subcomponent
function UserHeader() {
  return (
    <div className="header">
      <Avatar />
      <UserName />
    </div>
  );
}
```