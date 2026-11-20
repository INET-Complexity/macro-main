from abc import ABC, abstractmethod

import numpy as np


class ProductionSetter(ABC):
    """Abstract base class for determining firms' production processes.

    This class defines strategies for calculating production levels and
    input usage based on:
    - Production technology (Leontief, Linear, etc.)
    - Input availability and constraints
    - Input productivity and criticality
    - Utilization rates
    - Input substitution bundles

    The production process considers:
    - Labor inputs and constraints
    - Intermediate inputs (materials, supplies)
    - Capital inputs (machinery, equipment)
    - Input criticality and substitutability
    - Substitution between goods in the same bundle
    """

    def compute_production(
        self,
        desired_production: np.ndarray,
        current_labour_inputs: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
        tfp_multiplier: np.ndarray = None,
    ) -> np.ndarray:
        """Calculate actual production levels based on constraints.

        Determines feasible production by taking the minimum of:
        - Desired production targets
        - Available labor inputs (scaled by TFP)
        - Available intermediate and capital inputs (scaled by TFP)

        Args:
            desired_production (np.ndarray): Target production levels
            current_labour_inputs (np.ndarray): Available labor inputs
            current_limiting_intermediate_inputs (np.ndarray): Production
                possible with available intermediate inputs
            current_limiting_capital_inputs (np.ndarray): Production
                possible with available capital inputs
            tfp_multiplier (np.ndarray, optional): Total Factor Productivity
                multiplier for each firm. Defaults to 1.0 (no TFP effect)

        Returns:
            np.ndarray: Feasible production levels by firm
        """
        # Limiting stock is already TFP-scaled in set_targets()
        limiting_stock = self.compute_limiting_stock(
            current_limiting_intermediate_inputs,
            current_limiting_capital_inputs,
        )

        # Apply TFP multiplier to labour only (limiting stock is pre-scaled)
        if tfp_multiplier is not None:
            effective_labour = current_labour_inputs * tfp_multiplier
        else:
            effective_labour = current_labour_inputs

        return np.amin([desired_production, effective_labour, limiting_stock], axis=0)

    @abstractmethod
    def compute_limiting_intermediate_inputs_stock(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float | np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate production possible with available intermediate inputs.

        Determines maximum production levels considering:
        - Input-output coefficients
        - Available input stocks
        - Utilization rates
        - Input criticality
        - Substitution between goods in the same bundle

        Args:
            intermediate_inputs_productivity_matrix (np.ndarray): Input-output
                coefficients for intermediate inputs
            intermediate_inputs_stock (np.ndarray): Available stocks of
                intermediate inputs
            intermediate_inputs_utilisation_rate (float): Rate at which
                inputs can be utilized (0 to 1)
            goods_criticality_matrix (np.ndarray): Criticality levels
                for each input type
            substitution_bundle_matrix (np.ndarray): Matrix defining substitution
                bundles for goods, allowing substitution between goods in the same bundle

        Returns:
            np.ndarray: Maximum production possible with intermediate inputs
        """
        pass

    @abstractmethod
    def compute_limiting_capital_inputs_stock(
        self,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        capital_inputs_utilisation_rate: float | np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate production possible with available capital inputs.

        Determines maximum production levels considering:
        - Capital productivity coefficients
        - Available capital stocks
        - Utilization rates
        - Capital good criticality
        - Substitution between goods in the same bundle

        Args:
            capital_inputs_productivity_matrix (np.ndarray): Input-output
                coefficients for capital inputs
            capital_inputs_stock (np.ndarray): Available stocks of
                capital inputs
            capital_inputs_utilisation_rate (float): Rate at which
                capital can be utilized (0 to 1)
            goods_criticality_matrix (np.ndarray): Criticality levels
                for each capital good type
            substitution_bundle_matrix (np.ndarray): Matrix defining substitution
                bundles for goods, allowing substitution between goods in the same bundle

        Returns:
            np.ndarray: Maximum production possible with capital inputs
        """
        pass

    @staticmethod
    def compute_limiting_stock(
        limiting_intermediate_inputs_stock: np.ndarray,
        limiting_capital_inputs_stock: np.ndarray,
    ) -> np.ndarray:
        """Calculate overall production limits from input stocks.

        Takes the minimum of intermediate and capital input constraints
        to determine the binding constraint on production.

        Args:
            limiting_intermediate_inputs_stock (np.ndarray): Production
                possible with intermediate inputs
            limiting_capital_inputs_stock (np.ndarray): Production
                possible with capital inputs

        Returns:
            np.ndarray: Overall production limits from input constraints
        """
        return np.amin(
            [limiting_intermediate_inputs_stock, limiting_capital_inputs_stock],
            axis=0,
        )

    @abstractmethod
    def compute_intermediate_inputs_used(
        self,
        realised_production: np.ndarray,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate intermediate inputs consumed in production.

        Determines actual input usage based on:
        - Realized production levels
        - Input-output coefficients
        - Available stocks
        - Input criticality
        - Substitution between goods in the same bundle

        Args:
            realised_production (np.ndarray): Actual production achieved
            intermediate_inputs_productivity_matrix (np.ndarray): Input-output
                coefficients
            intermediate_inputs_stock (np.ndarray): Available input stocks
            goods_criticality_matrix (np.ndarray): Input criticality levels
            substitution_bundle_matrix (np.ndarray): Matrix defining substitution
                bundles for goods, allowing substitution between goods in the same bundle

        Returns:
            np.ndarray: Intermediate inputs used in production
        """
        pass

    @abstractmethod
    def compute_capital_inputs_used(
        self,
        realised_production: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate capital inputs consumed (depreciated) in production.

        Determines capital depreciation based on:
        - Realized production levels
        - Depreciation rates
        - Available capital stocks
        - Capital good criticality
        - Substitution between goods in the same bundle

        Args:
            realised_production (np.ndarray): Actual production achieved
            capital_inputs_depreciation_matrix (np.ndarray): Depreciation
                rates for capital goods
            capital_inputs_stock (np.ndarray): Available capital stocks
            goods_criticality_matrix (np.ndarray): Capital good criticality
            substitution_bundle_matrix (np.ndarray): Matrix defining substitution
                bundles for goods, allowing substitution between goods in the same bundle

        Returns:
            np.ndarray: Capital inputs depreciated in production
        """
        pass


class PureLeontief(ProductionSetter):
    """Implementation of pure Leontief production technology.

    This class implements a fixed-proportions production function where:
    - Inputs are perfectly complementary
    - No substitution between inputs is possible
    - Production is limited by the scarcest input
    - All inputs are treated as essential
    - Substitution bundles are not used (Leontief technology)
    """

    def compute_limiting_intermediate_inputs_stock(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float | np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate intermediate input limits under Leontief technology.

        Production is limited by the most constraining intermediate input,
        with no possibility of substitution.

        Args:
            intermediate_inputs_productivity_matrix (np.ndarray): Input-output
                coefficients for intermediate inputs
            intermediate_inputs_stock (np.ndarray): Available stocks of
                intermediate inputs
            intermediate_inputs_utilisation_rate (float): Rate at which
                inputs can be utilized (0 to 1)
            goods_criticality_matrix (np.ndarray): Criticality levels
                for each input type
            substitution_bundle_matrix (np.ndarray): Matrix defining substitution
                bundles for goods (not used in Leontief technology)

        Returns:
            np.ndarray: Production possible with intermediate inputs
        """

        output_mask = np.all((intermediate_inputs_productivity_matrix == np.inf), axis=1)

        limiting = np.multiply(
            intermediate_inputs_productivity_matrix,
            intermediate_inputs_stock,
            out=np.full(intermediate_inputs_productivity_matrix.shape, np.inf),
            where=intermediate_inputs_productivity_matrix != np.inf,
        ).min(axis=1)

        limiting[output_mask] = 0

        return limiting

    def compute_limiting_capital_inputs_stock(
        self,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        capital_inputs_utilisation_rate: float | np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate capital input limits under Leontief technology.

        Production is limited by the most constraining capital input,
        with no possibility of substitution.

        Args:
            capital_inputs_productivity_matrix (np.ndarray): Input-output
                coefficients for capital inputs
            capital_inputs_stock (np.ndarray): Available stocks of
                capital inputs
            capital_inputs_utilisation_rate (float): Rate at which
                capital can be utilized (0 to 1)
            goods_criticality_matrix (np.ndarray): Criticality levels
                for each capital good type
            substitution_bundle_matrix (np.ndarray): Matrix defining substitution
                bundles for goods (not used in Leontief technology)

        Returns:
            np.ndarray: Production possible with capital inputs
        """
        return np.multiply(
            capital_inputs_productivity_matrix,
            capital_inputs_stock,
            out=np.full(capital_inputs_productivity_matrix.shape, np.inf),
            where=capital_inputs_productivity_matrix != np.inf,
        ).min(axis=1)

    def compute_intermediate_inputs_used(
        self,
        realised_production: np.ndarray,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate intermediate inputs used under Leontief technology.

        Input usage is strictly proportional to production with fixed
        input-output coefficients.

        Args:
            realised_production (np.ndarray): Actual production achieved
            intermediate_inputs_productivity_matrix (np.ndarray): Input-output
                coefficients
            intermediate_inputs_stock (np.ndarray): Available input stocks
            goods_criticality_matrix (np.ndarray): Input criticality levels
            substitution_bundle_matrix (np.ndarray): Matrix defining substitution
                bundles for goods (not used in Leontief technology)

        Returns:
            np.ndarray: Intermediate inputs used in production
        """
        return np.divide(
            realised_production[:, None],
            intermediate_inputs_productivity_matrix,
            out=np.zeros_like(intermediate_inputs_productivity_matrix),
            where=intermediate_inputs_productivity_matrix != 0.0,
        )

    def compute_capital_inputs_used(
        self,
        realised_production: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate capital depreciation under Leontief technology.

        Capital depreciation is proportional to production with fixed
        depreciation rates.

        Args:
            realised_production (np.ndarray): Actual production achieved
            capital_inputs_depreciation_matrix (np.ndarray): Depreciation
                rates for capital goods
            capital_inputs_stock (np.ndarray): Available capital stocks
            goods_criticality_matrix (np.ndarray): Capital good criticality
            substitution_bundle_matrix (np.ndarray): Matrix defining substitution
                bundles for goods, allowing substitution between goods in the same bundle

        Returns:
            np.ndarray: Capital inputs depreciated in production
        """
        used_capital_inputs = realised_production[:, None] * capital_inputs_depreciation_matrix
        used_capital_inputs[used_capital_inputs == np.inf] = 0.0
        used_capital_inputs[used_capital_inputs == -np.inf] = 0.0
        return used_capital_inputs


class CriticalAndImportantLeontief(ProductionSetter):
    """Leontief production with critical and important input distinctions.

    This class extends the Leontief technology by:
    - Distinguishing between critical and non-critical inputs
    - Allowing production with only critical and important inputs
    - Maintaining fixed proportions within input categories
    - Treating non-critical inputs as optional
    """

    def compute_limiting_intermediate_inputs_stock(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float | np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate intermediate input limits with criticality.

        Only considers critical and important intermediate inputs as
        binding constraints on production.

        Args:
            [same as parent class]

        Returns:
            np.ndarray: Production possible with critical intermediate inputs
        """
        rescaled_intermediate_inputs = np.multiply(
            intermediate_inputs_productivity_matrix,
            intermediate_inputs_stock,
            out=np.full(intermediate_inputs_productivity_matrix.shape, np.inf),
            where=intermediate_inputs_productivity_matrix != np.inf,
        )
        rescaled_intermediate_inputs[goods_criticality_matrix == 0.0] = np.inf
        return rescaled_intermediate_inputs.min(axis=1)

    def compute_limiting_capital_inputs_stock(
        self,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        capital_inputs_utilisation_rate: float | np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate capital input limits with criticality.

        Only considers critical and important capital inputs as
        binding constraints on production.

        Args:
            [same as parent class]

        Returns:
            np.ndarray: Production possible with critical capital inputs
        """
        rescaled_capital_inputs = np.multiply(
            capital_inputs_productivity_matrix,
            capital_inputs_stock,
            out=np.full(capital_inputs_productivity_matrix.shape, np.inf),
            where=capital_inputs_productivity_matrix != np.inf,
        )
        rescaled_capital_inputs[goods_criticality_matrix == 0.0] = np.inf
        return rescaled_capital_inputs.min(axis=1)

    def compute_intermediate_inputs_used(
        self,
        realised_production: np.ndarray,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate intermediate inputs used with criticality.

        Uses inputs proportionally to production, but only for
        critical and important inputs.

        Args:
            [same as parent class]

        Returns:
            np.ndarray: Critical intermediate inputs used
        """
        used_intermediate_inputs = np.divide(
            realised_production[:, None],
            intermediate_inputs_productivity_matrix,
            out=np.zeros_like(intermediate_inputs_productivity_matrix),
            where=intermediate_inputs_productivity_matrix != 0.0,
        )
        used_intermediate_inputs[goods_criticality_matrix == 0.0] = 0.0
        return used_intermediate_inputs

    def compute_capital_inputs_used(
        self,
        realised_production: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate capital depreciation with criticality.

        Depreciates capital proportionally to production, but only for
        critical and important capital goods.

        Args:
            [same as parent class]

        Returns:
            np.ndarray: Critical capital inputs depreciated
        """
        used_capital_inputs = realised_production[:, None] * capital_inputs_depreciation_matrix
        used_capital_inputs[used_capital_inputs == np.inf] = 0.0
        used_capital_inputs[used_capital_inputs == -np.inf] = 0.0
        used_capital_inputs[goods_criticality_matrix == 0.0] = 0.0
        return used_capital_inputs


class CriticalLeontief(ProductionSetter):
    """Leontief production with only critical input constraints.

    This class implements a stricter version that:
    - Only considers fully critical inputs (criticality = 1)
    - Ignores all non-critical inputs in constraints
    - Maintains fixed proportions for critical inputs
    - Treats all other inputs as optional
    """

    def compute_limiting_intermediate_inputs_stock(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float | np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate intermediate input limits for critical inputs only.

        Only fully critical intermediate inputs (criticality = 1)
        are considered as binding constraints.

        Args:
            [same as parent class]

        Returns:
            np.ndarray: Production possible with critical inputs
        """
        rescaled_intermediate_inputs = np.multiply(
            intermediate_inputs_productivity_matrix,
            intermediate_inputs_stock,
            out=np.full(intermediate_inputs_productivity_matrix.shape, np.inf),
            where=intermediate_inputs_productivity_matrix != np.inf,
        )
        rescaled_intermediate_inputs[goods_criticality_matrix < 1.0] = np.inf
        return rescaled_intermediate_inputs.min(axis=1)

    def compute_limiting_capital_inputs_stock(
        self,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        capital_inputs_utilisation_rate: float | np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate capital input limits for critical inputs only.

        Only fully critical capital inputs (criticality = 1)
        are considered as binding constraints.

        Args:
            [same as parent class]

        Returns:
            np.ndarray: Production possible with critical capital
        """
        rescaled_capital_inputs = np.multiply(
            capital_inputs_productivity_matrix,
            capital_inputs_stock,
            out=np.full(capital_inputs_productivity_matrix.shape, np.inf),
            where=capital_inputs_productivity_matrix != np.inf,
        )
        rescaled_capital_inputs[goods_criticality_matrix < 1.0] = np.inf
        return rescaled_capital_inputs.min(axis=1)

    def compute_intermediate_inputs_used(
        self,
        realised_production: np.ndarray,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate critical intermediate inputs used.

        Uses inputs proportionally to production, but only for
        fully critical inputs.

        Args:
            [same as parent class]

        Returns:
            np.ndarray: Critical intermediate inputs used
        """
        used_intermediate_inputs = np.divide(
            realised_production[:, None],
            intermediate_inputs_productivity_matrix,
            out=np.zeros_like(intermediate_inputs_productivity_matrix),
            where=intermediate_inputs_productivity_matrix != 0.0,
        )
        used_intermediate_inputs[goods_criticality_matrix < 1.0] = 0.0
        return used_intermediate_inputs

    def compute_capital_inputs_used(
        self,
        realised_production: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate critical capital depreciation.

        Depreciates capital proportionally to production, but only for
        fully critical capital goods.

        Args:
            [same as parent class]

        Returns:
            np.ndarray: Critical capital inputs depreciated
        """
        used_capital_inputs = realised_production[:, None] * capital_inputs_depreciation_matrix
        used_capital_inputs[used_capital_inputs == np.inf] = 0.0
        used_capital_inputs[used_capital_inputs == -np.inf] = 0.0
        used_capital_inputs[goods_criticality_matrix < 1.0] = 0.0
        return used_capital_inputs


class Linear(ProductionSetter):
    """Implementation of linear production technology.

    This class implements a production function where:
    - Inputs are partially substitutable
    - Production scales linearly with inputs
    - Input utilization rates affect productivity
    - Criticality affects input requirements
    """

    def compute_limiting_intermediate_inputs_stock(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float | np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate intermediate input limits under linear technology.

        Production capacity scales linearly with input availability,
        adjusted by utilization rates.

        Args:
            [same as parent class]

        Returns:
            np.ndarray: Production possible with intermediate inputs
        """
        return np.multiply(
            intermediate_inputs_productivity_matrix,
            intermediate_inputs_stock,
            out=np.zeros_like(intermediate_inputs_productivity_matrix),
            where=intermediate_inputs_productivity_matrix != np.inf,
        ).sum(axis=1)

    def compute_limiting_capital_inputs_stock(
        self,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        capital_inputs_utilisation_rate: float | np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate capital input limits under linear technology.

        Production capacity scales linearly with capital availability,
        adjusted by utilization rates.

        Args:
            [same as parent class]

        Returns:
            np.ndarray: Production possible with capital inputs
        """
        return np.multiply(
            capital_inputs_productivity_matrix,
            capital_inputs_stock,
            out=np.zeros_like(capital_inputs_productivity_matrix),
            where=capital_inputs_productivity_matrix != np.inf,
        ).sum(axis=1)

    def compute_intermediate_inputs_used(
        self,
        realised_production: np.ndarray,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        total_used_intermediate_inputs = np.divide(
            realised_production[:, None],
            intermediate_inputs_productivity_matrix,
            out=np.zeros_like(intermediate_inputs_productivity_matrix),
            where=intermediate_inputs_productivity_matrix != 0.0,
        )
        used_intermediate_inputs = (
            total_used_intermediate_inputs.sum(axis=1)[:, None]
            * intermediate_inputs_stock
            / intermediate_inputs_stock.sum(axis=1, keepdims=True)
        )
        return used_intermediate_inputs

    def compute_capital_inputs_used(
        self,
        realised_production: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate capital depreciation under linear technology.

        Capital depreciation scales linearly with production,
        using depreciation coefficients.

        Args:
            [same as parent class]

        Returns:
            np.ndarray: Capital inputs depreciated in production
        """
        used_capital_inputs = realised_production[:, None] * capital_inputs_depreciation_matrix
        used_capital_inputs[used_capital_inputs == np.inf] = 0.0
        used_capital_inputs[used_capital_inputs == -np.inf] = 0.0
        used_capital_inputs = (
            used_capital_inputs.sum(axis=1)[:, None]
            * capital_inputs_stock
            / capital_inputs_stock.sum(axis=1, keepdims=True)
        )
        return used_capital_inputs


class UnconstrainedProduction(ProductionSetter):
    """Implementation of unconstrained production technology.

    This class implements a simplified production model where:
    - No input constraints are binding
    - All desired production is feasible
    - Input usage is tracked but not limiting
    - Useful for testing and special scenarios
    """

    def compute_limiting_intermediate_inputs_stock(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float | np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate intermediate input limits (unconstrained).

        Args:
            [same as parent class, all unused]

        Returns:
            np.ndarray: Production possible with intermediate inputs
        """
        return np.multiply(
            intermediate_inputs_productivity_matrix,
            intermediate_inputs_stock,
            out=np.full(intermediate_inputs_productivity_matrix.shape, np.inf),
            where=intermediate_inputs_productivity_matrix != np.inf,
        ).min(axis=1)

    def compute_limiting_capital_inputs_stock(
        self,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        capital_inputs_utilisation_rate: float | np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        return np.multiply(
            capital_inputs_productivity_matrix,
            capital_inputs_stock,
            out=np.full(capital_inputs_productivity_matrix.shape, np.inf),
            where=capital_inputs_productivity_matrix != np.inf,
        ).min(axis=1)

    def compute_intermediate_inputs_used(
        self,
        realised_production: np.ndarray,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        return np.zeros(intermediate_inputs_stock.shape)

    def compute_capital_inputs_used(
        self,
        realised_production: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate capital depreciation (unconstrained).

        Returns zero depreciation as production is unconstrained.

        Args:
            [same as parent class, all unused]

        Returns:
            np.ndarray: Zero capital depreciation
        """
        return np.zeros_like(capital_inputs_depreciation_matrix)

    def compute_production(
        self,
        desired_production: np.ndarray,
        current_labour_inputs: np.ndarray,
        current_limiting_intermediate_inputs: np.ndarray,
        current_limiting_capital_inputs: np.ndarray,
        tfp_multiplier: np.ndarray = None,
    ) -> np.ndarray:
        """Calculate production levels (unconstrained).

        Simply returns desired production as no constraints apply.

        Args:
            desired_production (np.ndarray): Target production levels
            current_labour_inputs (np.ndarray): Unused
            current_limiting_intermediate_inputs (np.ndarray): Unused
            current_limiting_capital_inputs (np.ndarray): Unused
            tfp_multiplier (np.ndarray, optional): Unused in unconstrained production

        Returns:
            np.ndarray: Desired production levels (unconstrained)
        """
        return desired_production


class BundledLeontief(ProductionSetter):
    """Implementation of Leontief production technology with substitution bundles.

    This class implements a Leontief production function where:
    - Inputs are perfectly complementary within each bundle
    - Substitution is possible between bundles
    - Production is limited by the scarcest input bundle
    - The substitution_bundle_matrix defines which inputs can be substituted for each other
    """

    def compute_limiting_intermediate_inputs_stock(
        self,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        intermediate_inputs_utilisation_rate: float | np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate intermediate input limits under bundled Leontief technology.

        Production is limited by the scarcest input bundle, with substitution
        possible between bundles.

        Args:
            intermediate_inputs_productivity_matrix (np.ndarray): Input-output
                coefficients for intermediate inputs
            intermediate_inputs_stock (np.ndarray): Available stocks of
                intermediate inputs
            intermediate_inputs_utilisation_rate (float): Rate at which
                inputs can be utilized (0 to 1)
            goods_criticality_matrix (np.ndarray): Criticality levels
                for each input type
            substitution_bundle_matrix (np.ndarray): Matrix defining substitution
                bundles for goods (shape: n_goods, n_bundles)

        Returns:
            np.ndarray: Production possible with intermediate inputs
        """
        # Calculate effective input availability
        effective_inputs = np.multiply(
            intermediate_inputs_productivity_matrix,
            intermediate_inputs_stock,
            out=np.full(intermediate_inputs_productivity_matrix.shape, np.inf),
            where=intermediate_inputs_productivity_matrix != np.inf,
        )

        # Apply substitution bundles
        # Multiply the effective inputs matrix with the substitution bundle matrix
        # to get bundle-level productivity
        # Note: substitution_bundle_matrix has shape (n_goods, n_bundles)
        bundle_productivity = np.matmul(effective_inputs, substitution_bundle_matrix)

        # industries with rows at infty are non-existent
        output_mask = np.all((intermediate_inputs_productivity_matrix == np.inf), axis=1)
        limiting = bundle_productivity.min(axis=1)
        limiting[output_mask] = 0

        # Take the minimum over bundles to get the limiting constraint
        return limiting

    def compute_limiting_capital_inputs_stock(
        self,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        capital_inputs_utilisation_rate: float | np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate capital input limits under bundled Leontief technology.

        Production is limited by the scarcest capital bundle, with substitution
        possible between bundles.

        Args:
            capital_inputs_productivity_matrix (np.ndarray): Input-output
                coefficients for capital inputs
            capital_inputs_stock (np.ndarray): Available stocks of
                capital inputs
            capital_inputs_utilisation_rate (float): Rate at which
                capital can be utilized (0 to 1)
            goods_criticality_matrix (np.ndarray): Criticality levels
                for each capital good type
            substitution_bundle_matrix (np.ndarray): Matrix defining substitution
                bundles for goods (shape: n_goods, n_bundles)

        Returns:
            np.ndarray: Production possible with capital inputs
        """
        # Calculate effective capital availability
        effective_capital = np.multiply(
            capital_inputs_productivity_matrix,
            capital_inputs_stock,
            out=np.full(capital_inputs_productivity_matrix.shape, np.inf),
            where=capital_inputs_productivity_matrix != np.inf,
        )

        # Apply substitution bundles
        # Multiply the effective capital matrix with the substitution bundle matrix
        # to get bundle-level productivity
        # Note: substitution_bundle_matrix has shape (n_goods, n_bundles)
        bundle_productivity = np.matmul(effective_capital, substitution_bundle_matrix)

        # Take the minimum over bundles to get the limiting constraint
        return bundle_productivity.min(axis=1)

    def compute_intermediate_inputs_used(
        self,
        realised_production: np.ndarray,
        intermediate_inputs_productivity_matrix: np.ndarray,
        intermediate_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate intermediate inputs used under bundled Leontief technology.

        Input usage is proportional to production with fixed input-output coefficients,
        adjusted by the substitution bundle matrix.

        Args:
            realised_production (np.ndarray): Actual production achieved
            intermediate_inputs_productivity_matrix (np.ndarray): Input-output
                coefficients
            intermediate_inputs_stock (np.ndarray): Available input stocks
            goods_criticality_matrix (np.ndarray): Input criticality levels
            substitution_bundle_matrix (np.ndarray): Matrix defining substitution
                bundles for goods (shape: n_goods, n_bundles)

        Returns:
            np.ndarray: Intermediate inputs used in production
        """
        # Calculate base input usage
        used_inputs = np.divide(
            realised_production[:, None],
            intermediate_inputs_productivity_matrix,
            out=np.zeros_like(intermediate_inputs_productivity_matrix),
            where=intermediate_inputs_productivity_matrix != 0.0,
        )

        # Apply substitution bundles to determine actual input usage
        # This is a simplified approach - in a full implementation,
        # you might want to optimize input usage across bundles
        return used_inputs

    def compute_capital_inputs_used(
        self,
        realised_production: np.ndarray,
        capital_inputs_depreciation_matrix: np.ndarray,
        capital_inputs_stock: np.ndarray,
        goods_criticality_matrix: np.ndarray,
        substitution_bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Calculate capital depreciation under bundled Leontief technology.

        Capital depreciation is proportional to production with fixed
        depreciation rates, adjusted by the substitution bundle matrix.

        Args:
            realised_production (np.ndarray): Actual production achieved
            capital_inputs_depreciation_matrix (np.ndarray): Depreciation
                rates for capital goods
            capital_inputs_stock (np.ndarray): Available capital stocks
            goods_criticality_matrix (np.ndarray): Capital good criticality
            substitution_bundle_matrix (np.ndarray): Matrix defining substitution
                bundles for goods (shape: n_goods, n_bundles)

        Returns:
            np.ndarray: Capital inputs depreciated in production
        """
        # Calculate base capital depreciation
        used_capital_inputs = realised_production[:, None] * capital_inputs_depreciation_matrix
        used_capital_inputs[used_capital_inputs == np.inf] = 0.0
        used_capital_inputs[used_capital_inputs == -np.inf] = 0.0

        # Apply substitution bundles to determine actual capital usage
        # This is a simplified approach - in a full implementation,
        # you might want to optimize capital usage across bundles
        return used_capital_inputs
